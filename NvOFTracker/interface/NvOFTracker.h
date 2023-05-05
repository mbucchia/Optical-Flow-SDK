/*
* This copyright notice applies to this header file only:
*
* Copyright (c) 2019-2022 NVIDIA Corporation
*
* Permission is hereby granted, free of charge, to any person
* obtaining a copy of this software and associated documentation
* files (the "Software"), to deal in the Software without
* restriction, including without limitation the rights to use,
* copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the software, and to permit persons to whom the
* software is furnished to do so, subject to the following
* conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
* OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
* NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
* HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
* WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
* OTHER DEALINGS IN THE SOFTWARE.
*/
/**
* \file NvOFTracker.h

*/


#pragma once
#include <stdlib.h>
#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

    typedef struct _NvOFT* NvOFTrackerHandle;

    /**
    *  Supported error codes
    */
    typedef enum _NvOFT_STATUS
    {
        /**
        * This indicates that API call returned with no errors.
        */
        NvOFT_SUCCESS = 0,

        /**
        * This indicates that HW Optical flow functionality is not supported
        */
        NvOFT_ERR_NvOF_NOT_SUPPORTED,

        /**
        * This indicates that one or more of the pointers passed to the API call
        * is invalid.
        */
        NvOFT_ERR_INVALID_PTR,

        /**
        * This indicates that one or more of the parameter passed to the API call
        * is invalid.
        */
        NvOFT_ERR_INVALID_PARAM,

        /**
        * This indicates that an API call was with an invalid handle.
        */
        NvOFT_ERR_INVALID_HANDLE,

        /**
        * This indicates that the API call failed because it was unable to allocate
        * enough system memory to perform the requested operation.
        */
        NvOFT_ERR_OUT_OF_SYSTEM_MEMORY,

        /**
        * This indicates that the API call failed because it was unable to allocate
        * enough video memory to perform the requested operation.
        */
        NvOFT_ERR_OUT_OF_VIDEO_MEMORY,

        /**
        * This indicates that the API call failed because openCV is not available on
        * system.
        */
        NvOFT_ERR_OPENCV_NOT_AVAILABLE,

        /**
        * This indicates that an unknown internal error has occurred.
        */
        NvOFT_ERR_GENERIC,
    }NvOFT_STATUS;
    
    /**
    * Defines memory types of input surface supported.
    */
    typedef enum _NvOFT_SURFACE_MEM_TYPE
    {
        NvOFT_SURFACE_MEM_TYPE_DEFAULT,         /**< Default = NvOFT_SURFACE_MEM_TYPE_CUDA_DEVPTR. */
        NvOFT_SURFACE_MEM_TYPE_SYSTEM,          /**< malloced memory. */
        NvOFT_SURFACE_MEM_TYPE_CUDA_DEVPTR,     /**< Surface type is CUdeviceptr. */
        NvOFT_SURFACE_MEM_TYPE_MAX,
    }NvOFT_SURFACE_MEM_TYPE;

    /**
    * Defines format of input surface supported.
    */
    typedef enum _NvOFT_SURFACE_FORMAT
    {
        NvOFT_SURFACE_FORMAT_DEFAULT,           /**< Default = NvOFT_SURFACE_FORMAT_Y. */
        NvOFT_SURFACE_FORMAT_Y,                 /**< Input surface format with 8 bit planar format. 
                                                     This is the preferred surface format.
                                                     If the surface format is different from the one's supported by NvOFT_SURFACE_FORMAT, client should convert it to NvOFT_SURFACE_FORMAT_Y. */
        NvOFT_SURFACE_FORMAT_NV12,              /**< Input surface format with 8 bit plannar, UV interleaved. */
        NvOFT_SURFACE_FORMAT_ABGR,              /**< Input surface format with ABGR-8-8-8-8 single plane. */
        NvOFT_SURFACE_FORMAT_MAX,
    }NvOFT_SURFACE_FORMAT;


    /**
    * \struct NvOFT_CREATE_PARAMS
    * NvOFTracker creation parameters.
    */
    typedef struct _NvOFT_CREATE_PARAMS
    {
        uint32_t width;                         /**< [in]: Specifies input surface width. */
        uint32_t height;                        /**< [in]: Specifies input surface height. */
        NvOFT_SURFACE_MEM_TYPE surfType;        /**< [in]: Specifies input surface memory type. */
        NvOFT_SURFACE_FORMAT surfFormat;        /**< [in]: Specifies input surface format. */
        uint32_t gpuID;                         /**< [in]: Specifies the gpu to be used. 
                                                           In case of a system with heterogeneouse GPU, client can loop through all the possible gpuID
                                                           to find out if NvOFTracker is supported on that GPU. */
        uint32_t reserved[32];
    }NvOFT_CREATE_PARAMS;

    /**
    * \struct NvOFT_SURFACE_PARAMS
    * NvOFTracker surface parameters.
    */
    typedef struct _NvOFT_SURFACE_PARAMS
    {
        uint32_t width;                         /**< [in]: Specifies input surface width. This value must be equal to NvOFT_CREATE_PARAMS::width."*/
        uint32_t height;                        /**< [in]: Specifies input surface height. This value must be equal to NvOFT_CREATE_PARAMS::height*/
        size_t pitch;                           /**< [in]: Specifies input surface pitch. */
        NvOFT_SURFACE_MEM_TYPE surfType;        /**< [in]: Specifies input surface memory type. */
        NvOFT_SURFACE_FORMAT surfFormat;        /**< [in]: Specifies input surface format. */
        const void* frameDataPtr;               /**< [in]: Pointer to the allocated surface. */
        size_t frameDataSize;                   /**< [in]: Size of allocated surface size in bytes.*/
        uint32_t reserved[32];
    }NvOFT_SURFACE_PARAMS;

    /**
    * \struct NvOFT_RECT
    * defines the rectangle.
    */
    typedef struct _NvOFT_RECT
    {
        int x;                                  /**< Pixel coordinate of left edge of the rectangle. */
        int y;                                  /**< Pixel coordinate of top edge of the rectangle. */
        int width;                              /**< Width of the rectangle in pixels. */
        int height;                             /**< Height of the rectangle in pixels. */
    }NvOFT_RECT;

    /**
    * \struct NvOFT_OBJ_TO_TRACK
    * information for each object to be tracked
    */
    typedef struct _NvOFT_OBJ_TO_TRACK
    {
        uint16_t classId;                       /**< [in]: Class of the object to be tracked. */
        NvOFT_RECT bbox;                        /**< [in]: Ojbect bounding box. */
        float confidence;                       /**< [in]: Detection confidence of the object. This is optional.*/
        void* pCustomData;                      /**< [in]: Used for the client to keep track of any data associated with the object. */
        uint32_t reserved[32];
    }NvOFT_OBJ_TO_TRACK;

    /**
    * \struct NvOFT_PROCESS_IN_PARAMS
    * Parameters for processing each frame
    * See NvOFTProcess
    */
    typedef struct _NvOFT_PROCESS_IN_PARAMS
    {
        uint32_t frameNum;                      /**< [in]: Frame number sequentially identifying the frame within a stream. This is optional. */
        bool detectionDone;                     /**< [in]: True if detection was done on this frame even if the list of
                                                           objects to track is empty. False otherwise. */
        bool reset;                             /**< [in]: True: reset tracking for the stream. */
        NvOFT_SURFACE_PARAMS surfParams;        /**< [in]: Surface parameters describing the current frame. */
        const NvOFT_OBJ_TO_TRACK* list;         /**< [in]: Pointer to a list/array of objects to be tracked in the current frame. */
        uint32_t listSize;                        /**< [in]: Number of populated blocks in the list. */
        uint32_t reserved[32];
    }NvOFT_PROCESS_IN_PARAMS;

    /**
    * \struct NvOFT_TRACKED_OBJ
    * Information for each tracked object
    */
    typedef struct _NvOFT_TRACKED_OBJ
    {
        uint16_t classId;                       /**< [out]: Class of the object to be tracked. */
        uint64_t trackingId;                    /**< [out]: Unique ID for the object as assigned by the tracker. */
        NvOFT_RECT bbox;                        /**< [out]: Bounding box. */
        float confidence;                       /**< [out]: Tracking confidence of the object. */
        uint32_t age;                           /**< [out]: Track length in frames. */
        NvOFT_OBJ_TO_TRACK* associatedObjectIn; /**< [out]: The associated input object if there is one. */
        uint32_t reserved[32];
    }NvOFT_TRACKED_OBJ;

    /**
    * \struct NvOFT_PROCESS_OUT_PARAMS
    * Parameters for processing each batch
    * See NvOFTProcess
    */
    typedef struct _NvOFT_PROCESS_OUT_PARAMS
    {
        uint32_t frameNum;                  /**< [out]: Frame number for objects in the list. */
        NvOFT_TRACKED_OBJ* list;            /**< [out]: Pointer to a list/array of object info blocks */
        uint32_t listSizeAllocated;           /**< [in]: Number of blocks allocated for the list. */
        uint32_t listSizeFilled;              /**< [out]: Number of populated blocks in the list. */
        uint32_t reserved[32];
    }NvOFT_PROCESS_OUT_PARAMS;

    /**
    * \brief Create an instance of NvOFTrackerHandle object.
    *
    * This function creates an instance of NvOFTrackerHandle object and returns status.
    * Client is expected to release NvOFTrackerHandle resource using NvOFTDestroy function call.
    *
    * \param [in] createParams
    *   Pointer of ::NvOFT_CREATE_PARAMS structure
    * \param [out] hOFT
    *   Pointer of ::NvOFTrackerHandle object.
    *
    * \return
    * ::NvOFT_SUCCESS \n
    * ::NvOFT_ERR_NvOF_NOT_SUPPORTED \n
    * ::NvOFT_ERR_INVALID_PTR \n
    * ::NvOFT_ERR_INVALID_PARAM \n
    * ::NvOFT_ERR_OUT_OF_SYSTEM_MEMORY \n
    * ::NvOFT_ERR_OUT_OF_VIDEO_MEMORY \n
    * ::NvOFT_ERR_OPENCV_NOT_AVAILABLE \n
    * ::NvOFT_ERR_GENERIC \n
    */
    NvOFT_STATUS NvOFTCreate(const NvOFT_CREATE_PARAMS* createParams, NvOFTrackerHandle* hOFT);

    /**
    * \brief Run NvOFTracker on the current frame.
    *
    * This function executes the NvOFTRacker on the current frame to track all the input objects 
    * In the absense of set of input objects for the current frame, it will track all the tracked objects from the previous frame.
    * The NvOFTProcess functions is supposed to be called every frame for better tracking accuracy.
    *
    * \param [in] hOFT
    *   ::NvOFTrackerHandle object.
    * \param [in] inParams
    *   Pointer of ::NvOFT_PROCESS_IN_PARAMS structure
    * \param [out] outParams
    *   Pointer of ::NvOFT_PROCESS_OUT_PARAMS structure
    *
    * \return
    * ::NvOFT_SUCCESS \n
    * ::NvOFT_ERR_INVALID_HANDLE \n
    * ::NvOFT_ERR_INVALID_PTR \n
    * ::NvOFT_ERR_INVALID_PARAM \n
    * ::NvOFT_ERR_OUT_OF_SYSTEM_MEMORY \n
    * ::NvOFT_ERR_OUT_OF_VIDEO_MEMORY \n
    * ::NvOFT_ERR_GENERIC \n
    */
    NvOFT_STATUS NvOFTProcess(NvOFTrackerHandle hOFT, const NvOFT_PROCESS_IN_PARAMS* inParams, NvOFT_PROCESS_OUT_PARAMS* outParams);

    /**
    * \brief Release NvOFTracker API  and resoures.
    *
    * Releases resources and waits until all resources are gracefully released.
    *
    *  \param [in] hOFT
    *   ::NvOFTrackerHandle object.
    *
    * \return
    * ::NvOFT_SUCCESS \n
    * ::NvOFT_ERR_INVALID_HANDLE \n
    * ::NvOFT_ERR_GENERIC \n
    */
    NvOFT_STATUS NvOFTDestroy(NvOFTrackerHandle hOFT);
#if defined(__cplusplus)
} // extern "C"
#endif/* __cplusplus */
