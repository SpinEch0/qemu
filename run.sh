sudo ./build/qemu-system-x86_64 -cpu host -enable-kvm -m 8g -smp 4 -name QEMU_1 \
         -drive file=/var/lib/libvirt/images/ubuntu24.04.qcow2,if=virtio,media=disk,cache=writeback,format=qcow2 \
-virtfs local,path=/mnt/shared,mount_tag=host0,security_model=passthrough,id=host0 \
-serial file:/home/zsj/serial.log \
-device pci-echodev \
-nic user,model=virtio-net-pci
#-nic tap





