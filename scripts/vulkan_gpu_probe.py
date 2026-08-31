#!/usr/bin/env python3
"""Enumerate Vulkan physical devices the way CARLA's ``-graphicsadapter`` does.

CARLA passes ``-graphicsadapter=N`` straight through to UE4's Vulkan RHI, which
indexes ``vkEnumeratePhysicalDevices`` output.  That order is *not* guaranteed to
match ``nvidia-smi``, so this probe prints the Vulkan index next to the PCI bus
ID (which ``nvidia-smi`` also reports) to give an exact mapping.

For each device it then runs a real GPU smoke test -- create a logical device,
allocate device-local memory, record a ``vkCmdFillBuffer``, submit it and wait on
a fence -- which is what surfaces ``VK_ERROR_DEVICE_LOST`` on a wedged card.

Standalone: stdlib only, no vulkan-tools and no root needed.
"""

import ctypes
import json
import os
import sys
from ctypes import POINTER, byref, c_char, c_char_p, c_float, c_uint8, c_uint32, c_uint64, c_void_p

# ---------------------------------------------------------------- constants

VK_SUCCESS = 0
VK_STRUCTURE_TYPE_APPLICATION_INFO = 0
VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1
VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO = 2
VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO = 3
VK_STRUCTURE_TYPE_SUBMIT_INFO = 4
VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO = 5
VK_STRUCTURE_TYPE_FENCE_CREATE_INFO = 8
VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO = 12
VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO = 39
VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO = 40
VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO = 42
VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 = 1000059001
VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT = 1000212000

VK_QUEUE_GRAPHICS_BIT = 0x1
VK_QUEUE_COMPUTE_BIT = 0x2
VK_BUFFER_USAGE_TRANSFER_DST_BIT = 0x2
VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT = 0x1
VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT = 0x1
VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT = 0x2
VK_MEMORY_HEAP_DEVICE_LOCAL_BIT = 0x1

VK_RESULT_NAMES = {
    0: 'VK_SUCCESS',
    2: 'VK_TIMEOUT',
    -1: 'VK_ERROR_OUT_OF_HOST_MEMORY',
    -2: 'VK_ERROR_OUT_OF_DEVICE_MEMORY',
    -3: 'VK_ERROR_INITIALIZATION_FAILED',
    -4: 'VK_ERROR_DEVICE_LOST',
    -5: 'VK_ERROR_MEMORY_MAP_FAILED',
    -6: 'VK_ERROR_LAYER_NOT_PRESENT',
    -7: 'VK_ERROR_EXTENSION_NOT_PRESENT',
    -8: 'VK_ERROR_FEATURE_NOT_PRESENT',
    -9: 'VK_ERROR_INCOMPATIBLE_DRIVER',
    -10: 'VK_ERROR_TOO_MANY_OBJECTS',
    -11: 'VK_ERROR_FORMAT_NOT_SUPPORTED',
    -1000069000: 'VK_ERROR_OUT_OF_POOL_MEMORY',
}

DEVICE_TYPE_NAMES = {0: 'OTHER', 1: 'INTEGRATED_GPU', 2: 'DISCRETE_GPU', 3: 'VIRTUAL_GPU', 4: 'CPU'}


def vkres(code):
    return VK_RESULT_NAMES.get(code, f'VkResult({code})')


# ---------------------------------------------------------------- structs


class VkApplicationInfo(ctypes.Structure):
    _fields_ = [
        ('sType', c_uint32),
        ('pNext', c_void_p),
        ('pApplicationName', c_char_p),
        ('applicationVersion', c_uint32),
        ('pEngineName', c_char_p),
        ('engineVersion', c_uint32),
        ('apiVersion', c_uint32),
    ]


class VkInstanceCreateInfo(ctypes.Structure):
    _fields_ = [
        ('sType', c_uint32),
        ('pNext', c_void_p),
        ('flags', c_uint32),
        ('pApplicationInfo', POINTER(VkApplicationInfo)),
        ('enabledLayerCount', c_uint32),
        ('ppEnabledLayerNames', POINTER(c_char_p)),
        ('enabledExtensionCount', c_uint32),
        ('ppEnabledExtensionNames', POINTER(c_char_p)),
    ]


class VkPhysicalDeviceProperties(ctypes.Structure):
    """Header fields spelled out; ``limits``/``sparseProperties`` kept opaque.

    The u64 tail forces the 8-byte alignment the real ``VkPhysicalDeviceLimits``
    has, so every header offset matches the driver's layout.  It is deliberately
    oversized (1 KiB vs ~0.5 KiB actual): extra room is harmless, a short buffer
    would be a driver-side overflow.
    """

    _fields_ = [
        ('apiVersion', c_uint32),
        ('driverVersion', c_uint32),
        ('vendorID', c_uint32),
        ('deviceID', c_uint32),
        ('deviceType', c_uint32),
        ('deviceName', c_char * 256),
        ('pipelineCacheUUID', c_uint8 * 16),
        ('_limits_and_sparse', c_uint64 * 128),
    ]


class VkPhysicalDeviceProperties2(ctypes.Structure):
    _fields_ = [('sType', c_uint32), ('pNext', c_void_p), ('properties', VkPhysicalDeviceProperties)]


class VkPhysicalDevicePCIBusInfoPropertiesEXT(ctypes.Structure):
    _fields_ = [
        ('sType', c_uint32),
        ('pNext', c_void_p),
        ('pciDomain', c_uint32),
        ('pciBus', c_uint32),
        ('pciDevice', c_uint32),
        ('pciFunction', c_uint32),
    ]


class VkExtent3D(ctypes.Structure):
    _fields_ = [('width', c_uint32), ('height', c_uint32), ('depth', c_uint32)]


class VkQueueFamilyProperties(ctypes.Structure):
    _fields_ = [
        ('queueFlags', c_uint32),
        ('queueCount', c_uint32),
        ('timestampValidBits', c_uint32),
        ('minImageTransferGranularity', VkExtent3D),
    ]


class VkMemoryType(ctypes.Structure):
    _fields_ = [('propertyFlags', c_uint32), ('heapIndex', c_uint32)]


class VkMemoryHeap(ctypes.Structure):
    _fields_ = [('size', c_uint64), ('flags', c_uint32)]


class VkPhysicalDeviceMemoryProperties(ctypes.Structure):
    _fields_ = [
        ('memoryTypeCount', c_uint32),
        ('memoryTypes', VkMemoryType * 32),
        ('memoryHeapCount', c_uint32),
        ('memoryHeaps', VkMemoryHeap * 16),
    ]


class VkDeviceQueueCreateInfo(ctypes.Structure):
    _fields_ = [
        ('sType', c_uint32),
        ('pNext', c_void_p),
        ('flags', c_uint32),
        ('queueFamilyIndex', c_uint32),
        ('queueCount', c_uint32),
        ('pQueuePriorities', POINTER(c_float)),
    ]


class VkDeviceCreateInfo(ctypes.Structure):
    _fields_ = [
        ('sType', c_uint32),
        ('pNext', c_void_p),
        ('flags', c_uint32),
        ('queueCreateInfoCount', c_uint32),
        ('pQueueCreateInfos', POINTER(VkDeviceQueueCreateInfo)),
        ('enabledLayerCount', c_uint32),
        ('ppEnabledLayerNames', POINTER(c_char_p)),
        ('enabledExtensionCount', c_uint32),
        ('ppEnabledExtensionNames', POINTER(c_char_p)),
        ('pEnabledFeatures', c_void_p),
    ]


class VkBufferCreateInfo(ctypes.Structure):
    _fields_ = [
        ('sType', c_uint32),
        ('pNext', c_void_p),
        ('flags', c_uint32),
        ('size', c_uint64),
        ('usage', c_uint32),
        ('sharingMode', c_uint32),
        ('queueFamilyIndexCount', c_uint32),
        ('pQueueFamilyIndices', POINTER(c_uint32)),
    ]


class VkMemoryRequirements(ctypes.Structure):
    _fields_ = [('size', c_uint64), ('alignment', c_uint64), ('memoryTypeBits', c_uint32)]


class VkMemoryAllocateInfo(ctypes.Structure):
    _fields_ = [
        ('sType', c_uint32),
        ('pNext', c_void_p),
        ('allocationSize', c_uint64),
        ('memoryTypeIndex', c_uint32),
    ]


class VkCommandPoolCreateInfo(ctypes.Structure):
    _fields_ = [('sType', c_uint32), ('pNext', c_void_p), ('flags', c_uint32), ('queueFamilyIndex', c_uint32)]


class VkCommandBufferAllocateInfo(ctypes.Structure):
    _fields_ = [
        ('sType', c_uint32),
        ('pNext', c_void_p),
        ('commandPool', c_uint64),
        ('level', c_uint32),
        ('commandBufferCount', c_uint32),
    ]


class VkCommandBufferBeginInfo(ctypes.Structure):
    _fields_ = [('sType', c_uint32), ('pNext', c_void_p), ('flags', c_uint32), ('pInheritanceInfo', c_void_p)]


class VkFenceCreateInfo(ctypes.Structure):
    _fields_ = [('sType', c_uint32), ('pNext', c_void_p), ('flags', c_uint32)]


class VkSubmitInfo(ctypes.Structure):
    _fields_ = [
        ('sType', c_uint32),
        ('pNext', c_void_p),
        ('waitSemaphoreCount', c_uint32),
        ('pWaitSemaphores', c_void_p),
        ('pWaitDstStageMask', c_void_p),
        ('commandBufferCount', c_uint32),
        ('pCommandBuffers', POINTER(c_void_p)),
        ('signalSemaphoreCount', c_uint32),
        ('pSignalSemaphores', c_void_p),
    ]


# ---------------------------------------------------------------- vulkan calls


def load_vulkan():
    for name in ('libvulkan.so.1', 'libvulkan.so'):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    raise SystemExit('could not load libvulkan.so.1 -- is the Vulkan loader installed?')


def create_instance(vk, api_version):
    app = VkApplicationInfo(
        sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,
        pApplicationName=b'carla-gpu-probe',
        applicationVersion=1,
        pEngineName=b'probe',
        engineVersion=1,
        apiVersion=api_version,
    )
    ci = VkInstanceCreateInfo(sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, pApplicationInfo=ctypes.pointer(app))
    inst = c_void_p()
    res = vk.vkCreateInstance(byref(ci), None, byref(inst))
    return res, inst


def enumerate_physical_devices(vk, inst):
    count = c_uint32(0)
    res = vk.vkEnumeratePhysicalDevices(inst, byref(count), None)
    if res != VK_SUCCESS:
        raise RuntimeError(f'vkEnumeratePhysicalDevices count failed: {vkres(res)}')
    arr = (c_void_p * max(count.value, 1))()
    res = vk.vkEnumeratePhysicalDevices(inst, byref(count), arr)
    if res != VK_SUCCESS:
        raise RuntimeError(f'vkEnumeratePhysicalDevices failed: {vkres(res)}')
    return [c_void_p(arr[i]) for i in range(count.value)]


def device_info(vk, phys):
    pci = VkPhysicalDevicePCIBusInfoPropertiesEXT(sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT)
    props2 = VkPhysicalDeviceProperties2(
        sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, pNext=ctypes.cast(byref(pci), c_void_p)
    )
    vk.vkGetPhysicalDeviceProperties2(phys, byref(props2))
    p = props2.properties

    mem = VkPhysicalDeviceMemoryProperties()
    vk.vkGetPhysicalDeviceMemoryProperties(phys, byref(mem))
    vram = max(
        (
            mem.memoryHeaps[i].size
            for i in range(mem.memoryHeapCount)
            if mem.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT
        ),
        default=0,
    )

    api, drv = p.apiVersion, p.driverVersion
    return {
        'device_name': p.deviceName.decode('utf-8', 'replace'),
        'device_type': DEVICE_TYPE_NAMES.get(p.deviceType, str(p.deviceType)),
        'vendor_id': f'0x{p.vendorID:04x}',
        'device_id': f'0x{p.deviceID:04x}',
        'vulkan_api': f'{(api >> 22) & 0x7F}.{(api >> 12) & 0x3FF}.{api & 0xFFF}',
        # NVIDIA packs its driver version as 10/8/8/6 bits, not the Vulkan 10/10/12.
        'nvidia_driver': f'{(drv >> 22) & 0x3FF}.{(drv >> 14) & 0xFF}',
        'vram_bytes': vram,
        'vram_gib': round(vram / (1024**3), 1),
        # Rendered the way nvidia-smi prints Bus-Id, so the two can be joined.
        'pci_bus_id': f'{pci.pciDomain:08X}:{pci.pciBus:02X}:{pci.pciDevice:02X}.{pci.pciFunction}',
    }


def pick_queue_family(vk, phys):
    count = c_uint32(0)
    vk.vkGetPhysicalDeviceQueueFamilyProperties(phys, byref(count), None)
    arr = (VkQueueFamilyProperties * max(count.value, 1))()
    vk.vkGetPhysicalDeviceQueueFamilyProperties(phys, byref(count), arr)
    for i in range(count.value):
        if arr[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT):
            return i, arr[i].queueFlags, arr[i].queueCount
    return None, 0, 0


def pick_memory_type(vk, phys, type_bits, required_flags):
    mem = VkPhysicalDeviceMemoryProperties()
    vk.vkGetPhysicalDeviceMemoryProperties(phys, byref(mem))
    for i in range(mem.memoryTypeCount):
        if (type_bits & (1 << i)) and (mem.memoryTypes[i].propertyFlags & required_flags) == required_flags:
            return i
    return None


def smoke_test(vk, phys, alloc_mib, fence_timeout_ns):
    """Create a device, fill a device-local buffer on the GPU, wait on a fence.

    Returns ``(ok, detail)``.  A wedged card typically fails here with
    ``VK_ERROR_DEVICE_LOST`` (or a fence timeout) even though enumeration and
    property queries succeeded -- those never touch the engine.
    """
    qf, flags, qcount = pick_queue_family(vk, phys)
    if qf is None:
        return False, 'no graphics/compute queue family'

    prio = (c_float * 1)(1.0)
    qci = VkDeviceQueueCreateInfo(
        sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, queueFamilyIndex=qf, queueCount=1, pQueuePriorities=prio
    )
    dci = VkDeviceCreateInfo(
        sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, queueCreateInfoCount=1, pQueueCreateInfos=ctypes.pointer(qci)
    )
    dev = c_void_p()
    res = vk.vkCreateDevice(phys, byref(dci), None, byref(dev))
    if res != VK_SUCCESS:
        return False, f'vkCreateDevice: {vkres(res)}'

    # Everything below is torn down in the finally block, in reverse order.
    buf, memory, pool, fence = c_uint64(0), c_uint64(0), c_uint64(0), c_uint64(0)
    try:
        queue = c_void_p()
        vk.vkGetDeviceQueue(dev, qf, 0, byref(queue))

        size = alloc_mib * 1024 * 1024
        bci = VkBufferCreateInfo(
            sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            size=size,
            usage=VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            sharingMode=0,
        )
        res = vk.vkCreateBuffer(dev, byref(bci), None, byref(buf))
        if res != VK_SUCCESS:
            return False, f'vkCreateBuffer: {vkres(res)}'

        req = VkMemoryRequirements()
        vk.vkGetBufferMemoryRequirements(dev, buf, byref(req))
        mt = pick_memory_type(vk, phys, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
        if mt is None:
            return False, 'no DEVICE_LOCAL memory type for buffer'

        mai = VkMemoryAllocateInfo(
            sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, allocationSize=req.size, memoryTypeIndex=mt
        )
        res = vk.vkAllocateMemory(dev, byref(mai), None, byref(memory))
        if res != VK_SUCCESS:
            return False, f'vkAllocateMemory({alloc_mib} MiB): {vkres(res)}'
        res = vk.vkBindBufferMemory(dev, buf, memory, c_uint64(0))
        if res != VK_SUCCESS:
            return False, f'vkBindBufferMemory: {vkres(res)}'

        cpci = VkCommandPoolCreateInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            queueFamilyIndex=qf,
        )
        res = vk.vkCreateCommandPool(dev, byref(cpci), None, byref(pool))
        if res != VK_SUCCESS:
            return False, f'vkCreateCommandPool: {vkres(res)}'

        cbai = VkCommandBufferAllocateInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, commandPool=pool, level=0, commandBufferCount=1
        )
        cmd = c_void_p()
        res = vk.vkAllocateCommandBuffers(dev, byref(cbai), byref(cmd))
        if res != VK_SUCCESS:
            return False, f'vkAllocateCommandBuffers: {vkres(res)}'

        bi = VkCommandBufferBeginInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
        )
        res = vk.vkBeginCommandBuffer(cmd, byref(bi))
        if res != VK_SUCCESS:
            return False, f'vkBeginCommandBuffer: {vkres(res)}'
        vk.vkCmdFillBuffer(cmd, buf, c_uint64(0), c_uint64(size), c_uint32(0xDEADBEEF))
        res = vk.vkEndCommandBuffer(cmd)
        if res != VK_SUCCESS:
            return False, f'vkEndCommandBuffer: {vkres(res)}'

        fci = VkFenceCreateInfo(sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO)
        res = vk.vkCreateFence(dev, byref(fci), None, byref(fence))
        if res != VK_SUCCESS:
            return False, f'vkCreateFence: {vkres(res)}'

        cmds = (c_void_p * 1)(cmd)
        si = VkSubmitInfo(sType=VK_STRUCTURE_TYPE_SUBMIT_INFO, commandBufferCount=1, pCommandBuffers=cmds)
        res = vk.vkQueueSubmit(queue, 1, byref(si), fence)
        if res != VK_SUCCESS:
            return False, f'vkQueueSubmit: {vkres(res)}'

        res = vk.vkWaitForFences(dev, 1, byref(fence), 1, c_uint64(fence_timeout_ns))
        if res != VK_SUCCESS:
            return False, f'vkWaitForFences: {vkres(res)}'

        res = vk.vkDeviceWaitIdle(dev)
        if res != VK_SUCCESS:
            return False, f'vkDeviceWaitIdle: {vkres(res)}'
        return True, f'queue_family={qf} flags=0x{flags:x} queues={qcount}; filled {alloc_mib} MiB device-local'
    finally:
        if fence.value:
            vk.vkDestroyFence(dev, fence, None)
        if pool.value:
            vk.vkDestroyCommandPool(dev, pool, None)
        if buf.value:
            vk.vkDestroyBuffer(dev, buf, None)
        if memory.value:
            vk.vkFreeMemory(dev, memory, None)
        vk.vkDestroyDevice(dev, None)


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--alloc-mib', type=int, default=64, help='device-local buffer to allocate and fill per GPU')
    ap.add_argument('--fence-timeout-s', type=float, default=20.0, help='how long to wait for the GPU submit')
    ap.add_argument('--json', metavar='PATH', help='also write results as JSON')
    ap.add_argument('--no-smoke-test', action='store_true', help='enumerate only, do not touch the GPUs')
    args = ap.parse_args()

    vk = load_vulkan()

    # Ask for 1.2 so vkGetPhysicalDeviceProperties2 is core; fall back to 1.0.
    res = None
    for api_name, api_version in (('1.2', (1 << 22) | (2 << 12)), ('1.0', 1 << 22)):
        res, inst = create_instance(vk, api_version)
        if res == VK_SUCCESS:
            break
    else:
        raise SystemExit(f'vkCreateInstance failed: {vkres(res)}')

    icd = os.environ.get('VK_ICD_FILENAMES', '<unset -- loader default discovery>')
    print(f'Vulkan instance created (requested API {api_name})')
    print(f'VK_ICD_FILENAMES = {icd}')

    physes = enumerate_physical_devices(vk, inst)
    print(f'vkEnumeratePhysicalDevices -> {len(physes)} device(s)\n')

    results = []
    for idx, phys in enumerate(physes):
        info = device_info(vk, phys)
        info['vulkan_index'] = idx
        if args.no_smoke_test:
            info['smoke_ok'], info['smoke_detail'] = None, 'skipped'
        else:
            ok, detail = smoke_test(vk, phys, args.alloc_mib, int(args.fence_timeout_s * 1e9))
            info['smoke_ok'], info['smoke_detail'] = ok, detail
        results.append(info)

        status = 'SKIPPED' if info['smoke_ok'] is None else ('OK' if info['smoke_ok'] else 'FAIL')
        print(f'[vulkan index {idx}]  ->  -graphicsadapter={idx}')
        print(f'    name       : {info["device_name"]} ({info["device_type"]})')
        print(f'    pci bus id : {info["pci_bus_id"]}')
        print(f'    vram       : {info["vram_gib"]} GiB device-local')
        print(f'    vulkan api : {info["vulkan_api"]}   nvidia driver: {info["nvidia_driver"]}')
        print(f'    smoke test : {status} -- {info["smoke_detail"]}')
        print()

    vk.vkDestroyInstance(inst, None)

    if args.json:
        with open(args.json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'wrote {args.json}')

    return 1 if any(r['smoke_ok'] is False for r in results) else 0


if __name__ == '__main__':
    sys.exit(main())
