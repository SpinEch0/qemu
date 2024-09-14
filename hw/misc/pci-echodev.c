#include "qemu/osdep.h"
#include "qemu/units.h"
#include "hw/pci/pci.h"
#include "hw/hw.h"
#include "hw/pci/msi.h"
#include "qemu/timer.h"
#include "qom/object.h"
#include "qemu/main-loop.h" /* iothread mutex */
#include "qemu/module.h"
#include "qapi/visitor.h"

#define TYPE_PCI_CUSTOM_DEVICE "pci-echodev"

#define ID_REGISTER        0x0
#define INV_REGISTER       0x4
#define IRQ_REGISTER       0x8
#define RANDVAL_REGISTER   0xc
#define DMA_SRC            0x10
#define DMA_DST            0x18
#define DMA_CNT            0x20
#define DMA_CMD            0x28

#define ATP_L              0x40
#define ATP_H              0x44

#define KERNEL_RING_BASE_L     0x48
#define KERNEL_RING_BASE_H     0x4c
#define KERNEL_RING_RPTR       0x50
#define KERNEL_RING_WPTR       0x54
#define KERNEL_RING_SIZE       0x58


#define IRQ_RING_BASE_L        0x60
#define IRQ_RING_BASE_H        0x64
#define IRQ_RING_RPTR          0x68
#define IRQ_RING_WPTR          0x6c
#define IRQ_RING_SIZE          0x70


#define BAR1_SIZE  (64 * 1024 * 1024)
#define PGSIZE 4096 // bytes per page
#define PGSHIFT 12  // bits of offset within a page

#define PGROUNDUP(sz)  (((sz)+PGSIZE-1) & ~(PGSIZE-1))
#define PGROUNDDOWN(a) (((a)) & ~(PGSIZE-1))

#define PTE_V (1L << 0) // valid
#define PTE_R (1L << 1)
#define PTE_W (1L << 2)
#define PTE_X (1L << 3)
#define PTE_U (1L << 4) // user can access

// shift a physical address to the right place for a PTE.
#define PA2PTE(pa) ((((uint64_t)pa) >> 12) << 10)

#define PTE2PA(pte) (((pte) >> 10) << 12)

#define PTE_FLAGS(pte) ((pte) & 0x3FF)

// extract the three 9-bit page table indices from a virtual address.
#define PXMASK          0x1FF // 9 bits
#define PXSHIFT(level)  (PGSHIFT+(9*(level)))
#define PX(level, va) ((((uint64_t) (va)) >> PXSHIFT(level)) & PXMASK)


typedef struct PciechodevState PciechodevState;
typedef uint64_t *pagetable_t; // 512 PTEs
typedef uint64_t pte_t;

//This macro provides the instance type cast functions for a QOM type.
DECLARE_INSTANCE_CHECKER(PciechodevState, PCIECHODEV, TYPE_PCI_CUSTOM_DEVICE)

//struct defining/descring the state
//of the custom pci device.
struct PciechodevState {
    PCIDevice pdev;
    MemoryRegion mmio_bar0;
    MemoryRegion mmio_bar1;
    uint32_t bar0[1024];
    uint8_t bar1[BAR1_SIZE];
    struct dma_state {
        dma_addr_t src;
        dma_addr_t dst;
        dma_addr_t cnt;
        dma_addr_t cmd;
    } *dma;
    QEMUTimer dma_timer;

    pagetable_t gpu_pagetable;
    QemuThread compute_thread;
    QemuMutex compute_mutex;
    QemuCond compute_cond;
    bool stopping;

};

#define DMA_RUN 1
#define DMA_DIR(cmd) (((cmd) & 2) >> 1)
#define DMA_TO_DEVICE 0
#define DMA_FROM_DEVICE 1
#define DMA_DONE (1<<31)
#define DMA_ERROR (1<<30)

static int check_range(uint64_t addr, uint64_t cnt)
{
    uint64_t end = addr + cnt;
    if(end > BAR1_SIZE)
        return -1;
    return 0;
}

// Return the address of the PTE in page table pagetable
// that corresponds to virtual address va.
//
// The risc-v Sv39 scheme has three levels of page-table
// pages. A page-table page contains 512 64-bit PTEs.
// A 64-bit virtual address is split into five fields:
//   39..63 -- must be zero.
//   30..38 -- 9 bits of level-2 index.
//   21..29 -- 9 bits of level-1 index.
//   12..20 -- 9 bits of level-0 index.
//    0..11 -- 12 bits of byte offset within the page.

static pte_t *walk(pagetable_t pagetable, uint64_t va)
{
    for(int level = 2; level > 0; level--) {
        pte_t *pte = &pagetable[PX(level, va)];
        if(*pte & PTE_V) {
            pagetable = (pagetable_t)PTE2PA(*pte);
        } else {
            return 0;
        }
    }
    return &pagetable[PX(0, va)];

}


// Look up a virtual address, return the physical address,
// or 0 if not mapped.
// Can only be used to look up user pages.
static uint64_t walkaddr(pagetable_t pagetable, uint64_t va)
{
  pte_t *pte;
  uint64_t pa;

  pte = walk(pagetable, va);
  if(pte == 0)
    return 0;
  if((*pte & PTE_V) == 0)
    return 0;
  if((*pte & PTE_U) == 0)
    return 0;
  pa = PTE2PA(*pte);
  return pa;
}


/*
 * Kernel launch, va is host virt data addr, gpu accesses this addr then triggers page fault and wait
 * gpu page table is saved in bar1[0]
 * driver migrates cpu memory to gpu memory in bar1 after gpu page table
 */
static void launch_kernel(PciechodevState *pdev, uint64_t va)
{
    int retry = 10;
    uint64_t gpu_pa = 0;
    char *gpu_data;
    uint64_t ih_ring_addr;
    uint32_t ih_ring_size;
    uint32_t ih_ring_rptr;
    uint32_t ih_ring_wptr;
    while(retry) {
        gpu_pa = walkaddr(pdev->gpu_pagetable, va);
        if (gpu_pa)
            break;
        if (retry == 10) {

            ih_ring_addr = pdev->bar0[IRQ_RING_BASE_H/4];
            ih_ring_addr = ih_ring_addr << 32;
            ih_ring_addr = pdev->bar0[IRQ_RING_BASE_L/4] | ih_ring_addr;

            ih_ring_size = pdev->bar0[IRQ_RING_SIZE/4];
            ih_ring_rptr = pdev->bar0[IRQ_RING_RPTR/4] & ih_ring_size;
            ih_ring_wptr = pdev->bar0[IRQ_RING_WPTR/4] & ih_ring_size;

            if (((ih_ring_rptr - ih_ring_wptr) & ih_ring_size) == 8) {
                printf("ih ring full!!!\n");
                sleep(10);
            }
            pci_dma_write(&pdev->pdev, ih_ring_addr + (ih_ring_wptr & ih_ring_size), &va, 8);
            pci_set_irq(&pdev->pdev, 1);
            pdev->bar0[IRQ_REGISTER/4] = 1;
            printf("raise gpu pagefault, ih ring %ld, rptr %d wptr %d\n", ih_ring_addr, ih_ring_rptr, ih_ring_wptr);
            pdev->bar0[IRQ_RING_WPTR/4] += 8;
        }
        retry--;
        sleep(1);
    }
    if (gpu_pa) {
        printf("cpu va to gpu pa\n");
        gpu_data = (void *)gpu_pa;
        gpu_data[0] = 1;
        gpu_data[1] = 2;
        gpu_data[2] = 3;
        gpu_data[4] = 4;
        gpu_data[5] = 3;
    } else {
        printf("page fault, kernel abort\n");
    }
}



/*
 * Compute thread, run gpu kernel
 */
static void *compute_thread(void *opaque)
{
    uint64_t kern_ring_addr;
    uint32_t kern_ring_size;
    uint32_t kern_ring_rptr;
    uint32_t kern_ring_wptr;
    PciechodevState *pdev = opaque;

    printf("echo dev compute thread start\n");
    while(1) {

        qemu_mutex_lock(&pdev->compute_mutex);
        qemu_cond_wait(&pdev->compute_cond, &pdev->compute_mutex);

        if (pdev->stopping) {
            qemu_mutex_unlock(&pdev->compute_mutex);
            break;
        }
        qemu_mutex_unlock(&pdev->compute_mutex);

        kern_ring_addr = pdev->bar0[KERNEL_RING_BASE_H/4];
        kern_ring_addr = kern_ring_addr << 32;
        kern_ring_addr = pdev->bar0[KERNEL_RING_BASE_L/4] | kern_ring_addr;

        kern_ring_size = pdev->bar0[KERNEL_RING_SIZE/4];
        kern_ring_rptr = pdev->bar0[KERNEL_RING_RPTR/4] & kern_ring_size;
        kern_ring_wptr = pdev->bar0[KERNEL_RING_WPTR/4] & kern_ring_size;

        // launch kernel
        printf("echo dev rptr %d wptr %d size %d\n", kern_ring_rptr, kern_ring_wptr, kern_ring_size);
        while (kern_ring_rptr != kern_ring_wptr) {
            uint64_t launch_addr;
            dma_addr_t kern_dma_addr = kern_ring_addr + (kern_ring_rptr & kern_ring_size);
            pci_dma_read(&pdev->pdev, kern_dma_addr,
            &launch_addr, 8);
            printf("cpu lauch addr %ld\n", launch_addr);
            launch_kernel(pdev, launch_addr);
            kern_ring_rptr += 8;
        }
        pdev->bar0[KERNEL_RING_RPTR/4] = kern_ring_rptr;
    }

    printf("echo dev compute thread end\n");
    return NULL;
}

static void fire_dma(PciechodevState *pciechodev)
{
    struct dma_state *dma = pciechodev->dma;
    dma->cmd &= ~(DMA_DONE | DMA_ERROR);

    if(DMA_DIR(dma->cmd) == DMA_TO_DEVICE) {
        printf("PCIECHODEV - Transfer Data from RC to EP\n");
        printf("pci_dma_read: src: %lx, dst: %lx, cnt: %ld, cmd: %lx\n",
            dma->src, dma->dst, dma->cnt, dma->cmd);
        if(check_range(dma->dst, dma->cnt) == 0) {
            pci_dma_read(&pciechodev->pdev, dma->src,
            pciechodev->bar1 + dma->dst, dma->cnt);
        } else
            dma->cmd |= (DMA_ERROR);
    } else {
        printf("PCIECHODEV - Transfer Data from EP to RC\n");
        printf("pci_dma_write: src: %lx, dst: %lx, cnt: %ld, cmd: %lx\n",
            dma->src, dma->dst, dma->cnt, dma->cmd);
        if(check_range(dma->src, dma->cnt) == 0) {
            pci_dma_write(&pciechodev->pdev, dma->dst,
            pciechodev->bar1 + dma->src, dma->cnt);
        } else
            dma->cmd |= (DMA_ERROR);
    }

    dma->cmd &= ~(DMA_RUN);
    dma->cmd |= (DMA_DONE);
    timer_mod(&pciechodev->dma_timer, qemu_clock_get_ms(QEMU_CLOCK_VIRTUAL) + 100);
}

static uint64_t pciechodev_bar0_mmio_read(void *opaque, hwaddr addr, unsigned size)
{
    PciechodevState *pciechodev = opaque;
    printf("PCIECHODEV: BAR0 pciechodev_mmio_read() addr %lx size %x \n", addr, size);

    if(addr == RANDVAL_REGISTER)
        return rand();

    return pciechodev->bar0[addr/4];
}

static void pciechodev_bar0_mmio_write(void *opaque, hwaddr addr, uint64_t val,
        unsigned size)
{
    printf("PCIECHODEV: BAR0 pciechodev_mmio_write() addr %lx size %x val %lx \n", addr, size, val);
    PciechodevState *pciechodev = opaque;

    switch(addr) {
        case ID_REGISTER:
        case RANDVAL_REGISTER:
            /* 0 and 12 are read only */
            break;
        case INV_REGISTER:
            pciechodev->bar0[1] = ~val;
            break;
        case DMA_CMD:
            pciechodev->dma->cmd = val;
            if(val & DMA_RUN)
                fire_dma(pciechodev);
            break;
        case IRQ_REGISTER:
            if(val & 1)
                pci_set_irq(&pciechodev->pdev, 1);
            else if(val & 2)
                pci_set_irq(&pciechodev->pdev, 0);
            pciechodev->bar0[addr/4] = val;
            break;
        case KERNEL_RING_WPTR:
            printf("kernel ring doorbell!!\n");
            pciechodev->bar0[addr/4] = val;
            qemu_cond_signal(&pciechodev->compute_cond);
            break;
        default:
            pciechodev->bar0[addr/4] = val;
            break;
    }
}

///ops for the Memory Region.
static const MemoryRegionOps pciechodev_bar0_mmio_ops = {
    .read = pciechodev_bar0_mmio_read,
    .write = pciechodev_bar0_mmio_write,
    .endianness = DEVICE_NATIVE_ENDIAN,
    .valid = {
        .min_access_size = 4,
        .max_access_size = 4,
    },
    .impl = {
        .min_access_size = 4,
        .max_access_size = 4,
    },

};

static uint64_t pciechodev_bar1_mmio_read(void *opaque, hwaddr addr, unsigned size)
{
    PciechodevState *pciechodev = opaque;
    printf("PCIECHODEV: BAR1 pciechodev_mmio_read() addr %lx size %x \n", addr, size);

    if(size == 1) {
        return pciechodev->bar1[addr];
    } else if(size == 2) {
        uint16_t *ptr = (uint16_t *) &pciechodev->bar1[addr];
        return *ptr;
    } else if(size == 4) {
        uint32_t *ptr = (uint32_t *) &pciechodev->bar1[addr];
        return *ptr;
    } else if(size == 8) {
        uint64_t *ptr = (uint64_t *) &pciechodev->bar1[addr];
        return *ptr;
    }
    return 0xffffffffffffffL;
}

static void pciechodev_bar1_mmio_write(void *opaque, hwaddr addr, uint64_t val,
        unsigned size)
{
    printf("PCIECHODEV: BAR1 pciechodev_mmio_read() addr %lx size %x val %lx \n", addr, size, val);
    PciechodevState *pciechodev = opaque;

    if(size == 1) {
        pciechodev->bar1[addr] = (uint8_t) val;
    } else if(size == 2) {
        uint16_t *ptr = (uint16_t *) &pciechodev->bar1[addr];
        *ptr = (uint16_t) val;
    } else if(size == 4) {
        uint32_t *ptr = (uint32_t *) &pciechodev->bar1[addr];
        *ptr = (uint32_t) val;
    } else if(size == 8) {
        uint64_t *ptr = (uint64_t *) &pciechodev->bar1[addr];
        *ptr = (uint64_t) val;
    }
}

///ops for the Memory Region.
static const MemoryRegionOps pciechodev_bar1_mmio_ops = {
    .read = pciechodev_bar1_mmio_read,
    .write = pciechodev_bar1_mmio_write,
    .endianness = DEVICE_NATIVE_ENDIAN,
    .valid = {
        .min_access_size = 1,
        .max_access_size = 8,
    },
    .impl = {
        .min_access_size = 1,
        .max_access_size = 8,
    },
};

static void edu_dma_timer(void *opaque)
{
    PciechodevState *pciechodev = opaque;
    pci_set_irq(&pciechodev->pdev, 1);
    pciechodev->bar0[IRQ_REGISTER/4] = 1;
}


//implementation of the realize function.
static void pci_pciechodev_realize(PCIDevice *pdev, Error **errp)
{
    PciechodevState *pciechodev = PCIECHODEV(pdev);
    uint8_t *pci_conf = pdev->config;

    pci_config_set_interrupt_pin(pci_conf, 1);

    ///initial configuration of devices registers.
    memset(pciechodev->bar0, 0, 4096);
    memset(pciechodev->bar1, 0, BAR1_SIZE);
    pciechodev->bar0[0] = 0xcafeaffe;
    pciechodev->dma = (struct dma_state *) &pciechodev->bar0[4];
    timer_init_ms(&pciechodev->dma_timer, QEMU_CLOCK_VIRTUAL, edu_dma_timer, pciechodev);

    // Initialize an I/O memory region(pciechodev->mmio).
    // Accesses to this region will cause the callbacks
    // of the pciechodev_mmio_ops to be called.
    memory_region_init_io(&pciechodev->mmio_bar0, OBJECT(pciechodev), &pciechodev_bar0_mmio_ops, pciechodev, "pciechodev-mmio", 4096);
    // registering the pdev and all of the above configuration
    // (actually filling a PCI-IO region with our configuration.
    pci_register_bar(pdev, 0, PCI_BASE_ADDRESS_SPACE_MEMORY, &pciechodev->mmio_bar0);
    /* BAR 1 */
    memory_region_init_io(&pciechodev->mmio_bar1, OBJECT(pciechodev), &pciechodev_bar1_mmio_ops, pciechodev, "pciechodev-mmio", BAR1_SIZE);
    pci_register_bar(pdev, 1, PCI_BASE_ADDRESS_SPACE_MEMORY, &pciechodev->mmio_bar1);

    qemu_mutex_init(&pciechodev->compute_mutex);
    qemu_cond_init(&pciechodev->compute_cond);

    pciechodev->stopping = false;
    pciechodev->gpu_pagetable = (pagetable_t)pciechodev->bar1;
    qemu_thread_create(&pciechodev->compute_thread, "echo compute thread",  compute_thread,
                       pciechodev, QEMU_THREAD_JOINABLE);
}

// uninitializing functions performed.
static void pci_pciechodev_uninit(PCIDevice *pdev)
{
    PciechodevState *pciechodev = PCIECHODEV(pdev);

    pciechodev->stopping = true;
    qemu_cond_signal(&pciechodev->compute_cond);
    qemu_thread_join(&pciechodev->compute_thread);
    qemu_cond_destroy(&pciechodev->compute_cond);
    qemu_mutex_destroy(&pciechodev->compute_mutex);

    timer_del(&pciechodev->dma_timer);
    return;
}


///initialization of the device
static void pciechodev_instance_init(Object *obj)
{
    return ;
}

static void pciechodev_class_init(ObjectClass *class, void *data)
{
    DeviceClass *dc = DEVICE_CLASS(class);
    PCIDeviceClass *k = PCI_DEVICE_CLASS(class);

    //definition of realize func().
    k->realize = pci_pciechodev_realize;
    //definition of uninit func().
    k->exit = pci_pciechodev_uninit;
    k->vendor_id = 0x8192;
    k->device_id = 0x0001; //our device id, 'beef' hexadecimal
    k->revision = 0x0;
    k->class_id = PCI_CLASS_OTHERS;

    /**
     * set_bit - Set a bit in memory
     * @nr: the bit to set
     * @addr: the address to start counting from
     */
    set_bit(DEVICE_CATEGORY_MISC, dc->categories);
}

static void pci_custom_device_register_types(void)
{
    static InterfaceInfo interfaces[] = {
        { INTERFACE_CONVENTIONAL_PCI_DEVICE },
        { },
    };
    static const TypeInfo custom_pci_device_info = {
        .name          = TYPE_PCI_CUSTOM_DEVICE,
        .parent        = TYPE_PCI_DEVICE,
        .instance_size = sizeof(PciechodevState),
        .instance_init = pciechodev_instance_init,
        .class_init    = pciechodev_class_init,
        .interfaces = interfaces,
    };
    //registers the new type.
    type_register_static(&custom_pci_device_info);
}

type_init(pci_custom_device_register_types)
