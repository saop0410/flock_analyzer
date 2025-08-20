#ifndef HAILO_INFOS_HPP_
#define HAILO_INFOS_HPP_

/*** Include ***/
/* for general */
#include <iostream>

/* for Hailo */
#include <hailo/hailort.hpp>

namespace hailo
{

inline static void print_stream_direction(hailo_stream_direction_t direction){
    switch (direction)
    {
    case HAILO_H2D_STREAM:
        std::cout << "direction : H2D, host to device" << std::endl;
        break;
    
    case HAILO_D2H_STREAM:
        std::cout << "direction : D2H, device to host" << std::endl;
        break;
    
    case HAILO_STREAM_DIRECTION_MAX_ENUM:
        std::cout << "direction : MAX_ENUM" << std::endl;
        break;
    
    default:
        std::cout << "direction : unknown" << std::endl;
        break;
    }
}

inline static void print_format_flags(hailo_format_flags_t flags){
    switch (flags)
    {
    case HAILO_FORMAT_FLAGS_NONE:
        std::cout << "format.flags : NONE" << std::endl;
        break;
    
    case HAILO_FORMAT_FLAGS_QUANTIZED:
        std::cout << "format.flags : QUANTIZED" << std::endl;
        break;
    
    case HAILO_FORMAT_FLAGS_TRANSPOSED:
        std::cout << "format.flags : TRANSPOSED" << std::endl;
        break;
    
    // case HAILO_FORMAT_FLAGS_HOST_ARGMAX:
    //     std::cout << "format.flags : HOST_ARGMAX, Only set on device side." << std::endl;
    //     break;
    
    case HAILO_FORMAT_FLAGS_MAX_ENUM:
        std::cout << "format.flags : MAX_ENUM" << std::endl;
        break;
    
    default:
        std::cout << "format.flags : unknown" << std::endl;
        break;
    }
}

inline static void print_format_order(hailo_format_order_t order){
    switch (order)
    {
    case HAILO_FORMAT_ORDER_AUTO:
        std::cout << "format.order : AUTO" << std::endl;
        break;
    
    case HAILO_FORMAT_ORDER_NHWC:
        std::cout << "format.order : NHWC, [N, H, W, C]" << std::endl;
        break;
    
    case HAILO_FORMAT_ORDER_NHCW:
        std::cout << "format.order : NHCW, [N, H, C, W]" << std::endl;
        break;
    
    case HAILO_FORMAT_ORDER_FCR:
        std::cout << "format.order : FCR, [N, H, W, C]" << std::endl;
        break;
    
    case HAILO_FORMAT_ORDER_F8CR:
        std::cout << "format.order : F8CR, [N, H, W, 8C], where channels are padded to 8 elements" << std::endl;
        break;
    
    case HAILO_FORMAT_ORDER_NHW:
        std::cout << "format.order : NHW, [N, H, W, 1]" << std::endl;
        break;
    
    case HAILO_FORMAT_ORDER_NC:
        std::cout << "format.order : NC, [N,C]" << std::endl;
        break;
    
    case HAILO_FORMAT_ORDER_BAYER_RGB:
        std::cout << "format.order : RGB, [N, H, W, 1]" << std::endl;
        break;
    
    case HAILO_FORMAT_ORDER_12_BIT_BAYER_RGB:
        std::cout << "format.order : 12_BIT_BAYER_RGB, [N, H, W, 1] where Channel is 12 bit" << std::endl;
        break;
    
    case HAILO_FORMAT_ORDER_HAILO_NMS:
        std::cout << "format.order : HAILO_NMS" << std::endl;
        break;
    
    case HAILO_FORMAT_ORDER_RGB888:
        std::cout << "format.order : RGB888, [N, H, W, C], where channels are 4 (RGB + 1 padded zero byte) and width is padded to 8 elements" << std::endl;
        break;
    
    case HAILO_FORMAT_ORDER_NCHW:
        std::cout << "format.order : NCHW, [N, C, H, W]" << std::endl;
        break;
    
    case HAILO_FORMAT_ORDER_YUY2:
        std::cout << "format.order : YUY2, [Y0, U0, Y1, V0]" << std::endl;
        break;
    
    case HAILO_FORMAT_ORDER_NV12:
        std::cout << "format.order : NV12, YUV format, encoding 8 pixels in 96 bits" << std::endl;
        break;
    
    case HAILO_FORMAT_ORDER_NV21:
        std::cout << "format.order : NV21, YUV format, encoding 8 pixels in 96 bits" << std::endl;
        break;
    
    case HAILO_FORMAT_ORDER_HAILO_YYUV:
        std::cout << "format.order : YYUV, Internal implementation for HAILO_FORMAT_ORDER_NV12 format" << std::endl;
        break;
    
    case HAILO_FORMAT_ORDER_HAILO_YYVU:
        std::cout << "format.order : YYVU, Internal implementation for HAILO_FORMAT_ORDER_NV21 format" << std::endl;
        break;
    
    case HAILO_FORMAT_ORDER_RGB4:
        std::cout << "format.order : RGB4, [N, H, W, C], where width*channels are padded to 4" << std::endl;
        break;
    
    case HAILO_FORMAT_ORDER_I420:
        std::cout << "format.order : I420,  YUV format, encoding 8 pixels in 96 bits" << std::endl;
        break;
    
    case HAILO_FORMAT_ORDER_HAILO_YYYYUV:
        std::cout << "format.order : YYYYUV, Internal implementation for HAILO_FORMAT_ORDER_I420 format" << std::endl;
        break;
    
    case HAILO_FORMAT_ORDER_HAILO_NMS_WITH_BYTE_MASK:
        std::cout << "format.order : NMS_WITH_BYTE_MASK" << std::endl;
        break;
    
    case HAILO_FORMAT_ORDER_MAX_ENUM:
        std::cout << "format.order : MAX_ENUM" << std::endl;
        break;
    
    default:
        std::cout << "format.order : unknown" << std::endl;
        break;
    }
}

inline void print_format_type(hailo_format_type_t type){
    switch (type)
    {
    case HAILO_FORMAT_TYPE_AUTO:
        std::cout << "format.type  : AUTO" << std::endl;
        break;
    
    case HAILO_FORMAT_TYPE_UINT8:
        std::cout << "format.type  : UINT8" << std::endl;
        break;
    
    case HAILO_FORMAT_TYPE_UINT16:
        std::cout << "format.type  : UINT16" << std::endl;
        break;
    
    case HAILO_FORMAT_TYPE_FLOAT32:
        std::cout << "format.type  : FLOAT32" << std::endl;
        break;
    
    case HAILO_FORMAT_TYPE_MAX_ENUM:
        std::cout << "format.type  : MAX_ENUM" << std::endl;
        break;
    
    default:
        std::cout << "format.type : unknown" << std::endl;
        break;
    }
}

inline static void print_vstream_info(std::vector<hailo_vstream_info_t> infos) {
    size_t count = 0;
    for (auto info : infos) {
        std::cout << "======== batch " << count  << " =========" << std::endl;
        std::cout << "name : " << info.name << std::endl;
        std::cout << "network_name : " << info.network_name << std::endl;
        print_stream_direction(info.direction);
        print_format_flags(info.format.flags);
        print_format_order(info.format.order);
        print_format_type(info.format.type);
        std::cout << "shape.features : " << info.shape.features << std::endl;
        std::cout << "shape.height   : " << info.shape.height << std::endl;
        std::cout << "shape.width    : " << info.shape.width << std::endl;
        std::cout << "nms_info.max_bboxes_per_class : " << info.nms_shape.max_bboxes_per_class << std::endl;
        // std::cout << "nms_info.max_mask_size        : " << info.nms_shape.max_mask_size << std::endl;
        std::cout << "nms_info.number_of_classes    : " << info.nms_shape.number_of_classes << std::endl;
        std::cout << "quant_info.limvals_max : " << info.quant_info.limvals_max << std::endl;
        std::cout << "quant_info.limvals_min : " << info.quant_info.limvals_min << std::endl;
        std::cout << "quant_info.qp_scale    : " << info.quant_info.qp_scale << std::endl;
        std::cout << "quant_info.qp_zp       : " << info.quant_info.qp_zp << std::endl;
        count++;
    }
}

}

#endif