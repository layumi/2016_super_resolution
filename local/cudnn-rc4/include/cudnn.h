/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

 /*   cudnn : Neural Networks Library

 */

#if !defined(CUDNN_H_)
#define CUDNN_H_

#define CUDNN_MAJOR      4
#define CUDNN_MINOR      0
#define CUDNN_PATCHLEVEL 4

#define CUDNN_VERSION    (CUDNN_MAJOR * 1000 + CUDNN_MINOR * 100 + CUDNN_PATCHLEVEL)

#include "driver_types.h"
#include <cuda_runtime.h>

#ifndef CUDNNWINAPI
#ifdef _WIN32
#define CUDNNWINAPI __stdcall
#else
#define CUDNNWINAPI
#endif
#endif

#if defined (__cplusplus)
extern "C" {
#endif

struct cudnnContext;
typedef struct cudnnContext *cudnnHandle_t;

size_t CUDNNWINAPI cudnnGetVersion(void);

/*
 * CUDNN return codes
 */
typedef enum
{
    CUDNN_STATUS_SUCCESS          = 0,
    CUDNN_STATUS_NOT_INITIALIZED  = 1,
    CUDNN_STATUS_ALLOC_FAILED     = 2,
    CUDNN_STATUS_BAD_PARAM        = 3,
    CUDNN_STATUS_INTERNAL_ERROR   = 4,
    CUDNN_STATUS_INVALID_VALUE    = 5,
    CUDNN_STATUS_ARCH_MISMATCH    = 6,
    CUDNN_STATUS_MAPPING_ERROR    = 7,
    CUDNN_STATUS_EXECUTION_FAILED = 8,
    CUDNN_STATUS_NOT_SUPPORTED    = 9,
    CUDNN_STATUS_LICENSE_ERROR    = 10
} cudnnStatus_t;

// human-readable error messages
const char *  CUDNNWINAPI cudnnGetErrorString(cudnnStatus_t status);

cudnnStatus_t CUDNNWINAPI cudnnCreate        (cudnnHandle_t *handle);
cudnnStatus_t CUDNNWINAPI cudnnDestroy       (cudnnHandle_t handle);
cudnnStatus_t CUDNNWINAPI cudnnSetStream     (cudnnHandle_t handle, cudaStream_t streamId);
cudnnStatus_t CUDNNWINAPI cudnnGetStream     (cudnnHandle_t handle, cudaStream_t *streamId);


/* Data structures to represent Image/Filter and the Neural Network Layer */
typedef struct cudnnTensorStruct*        cudnnTensorDescriptor_t;
typedef struct cudnnConvolutionStruct*   cudnnConvolutionDescriptor_t;
typedef struct cudnnPoolingStruct*       cudnnPoolingDescriptor_t;
typedef struct cudnnFilterStruct*        cudnnFilterDescriptor_t;
typedef struct cudnnLRNStruct*           cudnnLRNDescriptor_t;
typedef struct cudnnActivationStruct*    cudnnActivationDescriptor_t;
/*
* CUDNN data type
*/
typedef enum
{
    CUDNN_DATA_FLOAT  = 0,
    CUDNN_DATA_DOUBLE = 1,
    CUDNN_DATA_HALF   = 2,
} cudnnDataType_t;

/*
 * CUDNN propagate Nan
 */
typedef enum{
    CUDNN_NOT_PROPAGATE_NAN  = 0,
    CUDNN_PROPAGATE_NAN      = 1,
} cudnnNanPropagation_t;

/* Maximum supported number of tensor dimensions */
#define CUDNN_DIM_MAX 8

/* Create an instance of a generic Tensor descriptor */
cudnnStatus_t CUDNNWINAPI cudnnCreateTensorDescriptor(
                                cudnnTensorDescriptor_t            *tensorDesc );

typedef enum
{
    CUDNN_TENSOR_NCHW = 0,   /* row major (wStride = 1, hStride = w) */
    CUDNN_TENSOR_NHWC = 1    /* feature maps interleaved ( cStride = 1 )*/
} cudnnTensorFormat_t;

cudnnStatus_t CUDNNWINAPI cudnnSetTensor4dDescriptor(
                                cudnnTensorDescriptor_t             tensorDesc,
                                cudnnTensorFormat_t                 format,
                                cudnnDataType_t                     dataType, // image data type
                                int                                 n,        // number of inputs (batch size)
                                int                                 c,        // number of input feature maps
                                int                                 h,        // height of input section
                                int                                 w );       // width of input section


cudnnStatus_t CUDNNWINAPI cudnnSetTensor4dDescriptorEx(
                                cudnnTensorDescriptor_t             tensorDesc,
                                cudnnDataType_t                     dataType, // image data type
                                int                                 n,        // number of inputs (batch size)
                                int                                 c,        // number of input feature maps
                                int                                 h,        // height of input section
                                int                                 w,        // width of input section
                                int                                 nStride,
                                int                                 cStride,
                                int                                 hStride,
                                int                                 wStride );

cudnnStatus_t CUDNNWINAPI cudnnGetTensor4dDescriptor(
                                const cudnnTensorDescriptor_t       tensorDesc,
                                cudnnDataType_t                    *dataType, // image data type
                                int                                *n,        // number of inputs (batch size)
                                int                                *c,        // number of input feature maps
                                int                                *h,        // height of input section
                                int                                *w,        // width of input section
                                int                                *nStride,
                                int                                *cStride,
                                int                                *hStride,
                                int                                *wStride );

cudnnStatus_t CUDNNWINAPI cudnnSetTensorNdDescriptor(
                                cudnnTensorDescriptor_t             tensorDesc,
                                cudnnDataType_t                     dataType,
                                int                                 nbDims,
                                const int                           dimA[],
                                const int                           strideA[] );

cudnnStatus_t CUDNNWINAPI cudnnGetTensorNdDescriptor(
                                const cudnnTensorDescriptor_t       tensorDesc,
                                int                                 nbDimsRequested,
                                cudnnDataType_t                    *dataType,
                                int                                *nbDims,
                                int                                 dimA[],
                                int                                 strideA[] );

/* PixelOffset( n, c, h, w ) = n *input_stride + c * feature_stride + h * h_stride + w * w_stride

   1)Example of all images in row major order one batch of features after the other (with an optional padding on row)
   input_stride :  c x h x h_stride
   feature_stride : h x h_stride
   h_stride  :  >= w  ( h_stride = w if no padding)
   w_stride  : 1


   2)Example of all images in row major with features maps interleaved
   input_stride :  c x h x h_stride
   feature_stride : 1
   h_stride  :  w x c
   w_stride  : c

   3)Example of all images in column major order one batch of features after the other (with optional padding on column)
   input_stride :  c x w x w_stride
   feature_stride : w x w_stride
   h_stride  :  1
   w_stride  :  >= h

*/

/* Destroy an instance of Tensor4d descriptor */
cudnnStatus_t CUDNNWINAPI cudnnDestroyTensorDescriptor(
                                cudnnTensorDescriptor_t             tensorDesc );


/* Tensor layout conversion helper (y = alpha * x + beta * y) */
cudnnStatus_t CUDNNWINAPI cudnnTransformTensor(
                                cudnnHandle_t                       handle,
                                const void                         *alpha,
                                const cudnnTensorDescriptor_t       xDesc,
                                const void                         *x,
                                const void                         *beta,
                                const cudnnTensorDescriptor_t       yDesc,
                                void                               *y );

typedef enum
{
   /* add one image to every feature maps of each input */
   CUDNN_ADD_IMAGE   = 0,
   CUDNN_ADD_SAME_HW = 0,

   /* add a set of feature maps to a batch of inputs : tensorBias has n=1 , same number of features as x and y */
   CUDNN_ADD_FEATURE_MAP = 1,
   CUDNN_ADD_SAME_CHW    = 1,

   /* add a tensor of size 1,c,1,1 to every corresponding point of n,c,h,w input */
   CUDNN_ADD_SAME_C      = 2,

   /* add 2 tensors with same n,c,h,w */
   CUDNN_ADD_FULL_TENSOR = 3
} cudnnAddMode_t;

/* Tensor Bias addition : y = alpha * b + beta * y  */
cudnnStatus_t CUDNNWINAPI cudnnAddTensor(
                                cudnnHandle_t                       handle,
                                const void                         *alpha,
                                const cudnnTensorDescriptor_t       bDesc,
                                const void                         *b,
                                const void                         *beta,
                                cudnnTensorDescriptor_t             yDesc,
                                void                               *y );

/* cudnnAddTensor_v3 is now mapped to cudnnAddTensor
   and will be removed at the same time as cudnnAddTensor_v2
   Use cudnnAddTensor instead
 */
cudnnStatus_t CUDNNWINAPI cudnnAddTensor_v3(
                                cudnnHandle_t                       handle,
                                const void                         *alpha,
                                const cudnnTensorDescriptor_t       bDesc,
                                const void                         *b,
                                const void                         *beta,
                                cudnnTensorDescriptor_t             yDesc,
                                void                               *y );

/* Set all values of a tensor to a given value : y[i] = value[0] */
cudnnStatus_t CUDNNWINAPI cudnnSetTensor(
                                cudnnHandle_t                       handle,
                                const cudnnTensorDescriptor_t       yDesc,
                                void                               *y,
                                const void                         *valuePtr );

/* Scale all values of a tensor by a given factor : y[i] = alpha * y[i] */
cudnnStatus_t CUDNNWINAPI cudnnScaleTensor(
                                cudnnHandle_t                       handle,
                                const cudnnTensorDescriptor_t       yDesc,
                                void                               *y,
                                const void                         *alpha );

/*
 *  convolution mode
 */
typedef enum
{
    CUDNN_CONVOLUTION       = 0,
    CUDNN_CROSS_CORRELATION = 1
} cudnnConvolutionMode_t;


/* Create an instance of FilterStruct */
cudnnStatus_t CUDNNWINAPI cudnnCreateFilterDescriptor(
                                cudnnFilterDescriptor_t            *filterDesc );

cudnnStatus_t CUDNNWINAPI cudnnSetFilter4dDescriptor(
                                cudnnFilterDescriptor_t             filterDesc,
                                cudnnDataType_t                     dataType, // image data type
                                int                                 k,        // number of output feature maps
                                int                                 c,        // number of input feature maps
                                int                                 h,        // height of each input filter
                                int                                 w );      // width of  each input filter

cudnnStatus_t CUDNNWINAPI cudnnSetFilter4dDescriptor_v4(
                                cudnnFilterDescriptor_t             filterDesc,
                                cudnnDataType_t                     dataType, // image data type
                                cudnnTensorFormat_t                 format,
                                int                                 k,        // number of output feature maps
                                int                                 c,        // number of input feature maps
                                int                                 h,        // height of each input filter
                                int                                 w );      // width of  each input filter

cudnnStatus_t CUDNNWINAPI cudnnGetFilter4dDescriptor(
                                const cudnnFilterDescriptor_t       filterDesc,
                                cudnnDataType_t                    *dataType, // image data type
                                int                                *k,        // number of output feature maps
                                int                                *c,        // number of input feature maps
                                int                                *h,        // height of each input filter
                                int                                *w );      // width of  each input filter

cudnnStatus_t CUDNNWINAPI cudnnGetFilter4dDescriptor_v4(
                                const cudnnFilterDescriptor_t       filterDesc,
                                cudnnDataType_t                    *dataType, // image data type
                                cudnnTensorFormat_t                *format,
                                int                                *k,        // number of output feature maps
                                int                                *c,        // number of input feature maps
                                int                                *h,        // height of each input filter
                                int                                *w );      // width of  each input filter

cudnnStatus_t CUDNNWINAPI cudnnSetFilterNdDescriptor(
                                cudnnFilterDescriptor_t             filterDesc,
                                cudnnDataType_t                     dataType, // image data type
                                int                                 nbDims,
                                const int                           filterDimA[] );


cudnnStatus_t CUDNNWINAPI cudnnSetFilterNdDescriptor_v4(
                                cudnnFilterDescriptor_t             filterDesc,
                                cudnnDataType_t                     dataType, // image data type
                                cudnnTensorFormat_t                 format,
                                int                                 nbDims,
                                const int                           filterDimA[] );

cudnnStatus_t CUDNNWINAPI cudnnGetFilterNdDescriptor(
                                const cudnnFilterDescriptor_t       filterDesc,
                                int                                 nbDimsRequested,
                                cudnnDataType_t                    *dataType, // image data type
                                int                                *nbDims,
                                int                                 filterDimA[] );

cudnnStatus_t CUDNNWINAPI cudnnGetFilterNdDescriptor_v4(
                                const cudnnFilterDescriptor_t       filterDesc,
                                int                                 nbDimsRequested,
                                cudnnDataType_t                    *dataType, // image data type
                                cudnnTensorFormat_t                *format,
                                int                                *nbDims,
                                int                                 filterDimA[] );

cudnnStatus_t CUDNNWINAPI cudnnDestroyFilterDescriptor(
                                cudnnFilterDescriptor_t             filterDesc );

/* Create an instance of convolution descriptor */
cudnnStatus_t CUDNNWINAPI cudnnCreateConvolutionDescriptor(
                                cudnnConvolutionDescriptor_t       *convDesc );

cudnnStatus_t CUDNNWINAPI cudnnSetConvolution2dDescriptor(
                                cudnnConvolutionDescriptor_t        convDesc,
                                int                                 pad_h,    // zero-padding height
                                int                                 pad_w,    // zero-padding width
                                int                                 u,        // vertical filter stride
                                int                                 v,        // horizontal filter stride
                                int                                 upscalex, // upscale the input in x-direction
                                int                                 upscaley, // upscale the input in y-direction
                                cudnnConvolutionMode_t              mode );


cudnnStatus_t CUDNNWINAPI cudnnGetConvolution2dDescriptor(
                                const cudnnConvolutionDescriptor_t  convDesc,
                                int                                *pad_h,    // zero-padding height
                                int                                *pad_w,    // zero-padding width
                                int                                *u,        // vertical filter stride
                                int                                *v,        // horizontal filter stride
                                int                                *upscalex, // upscale the input in x-direction
                                int                                *upscaley, // upscale the input in y-direction
                                cudnnConvolutionMode_t             *mode );

/* Helper function to return the dimensions of the output tensor given a convolution descriptor */
cudnnStatus_t CUDNNWINAPI cudnnGetConvolution2dForwardOutputDim(
                                const cudnnConvolutionDescriptor_t  convDesc,
                                const cudnnTensorDescriptor_t       inputTensorDesc,
                                const cudnnFilterDescriptor_t       filterDesc,
                                int                                *n,
                                int                                *c,
                                int                                *h,
                                int                                *w );


cudnnStatus_t CUDNNWINAPI cudnnSetConvolutionNdDescriptor(
                                cudnnConvolutionDescriptor_t        convDesc,
                                int                                 arrayLength,             /* nbDims-2 size */
                                const int                           padA[],
                                const int                           filterStrideA[],
                                const int                           upscaleA[],
                                cudnnConvolutionMode_t              mode,
                                cudnnDataType_t                     dataType );  // convolution data type

cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionNdDescriptor(
                                const cudnnConvolutionDescriptor_t  convDesc,
                                int                                 arrayLengthRequested,
                                int                                *arrayLength,
                                int                                 padA[],
                                int                                 strideA[],
                                int                                 upscaleA[],
                                cudnnConvolutionMode_t             *mode,
                                cudnnDataType_t                    *dataType );   // convolution data type

/* cudnnSetConvolutionNdDescriptor_v3 is now mapped to cudnnSetConvolutionNdDescriptor
   and will be removed at the same time than cudnnSetConvolutionNdDescriptor_v2
   Use cudnnSetConvolutionNdDescriptor instead */
cudnnStatus_t CUDNNWINAPI cudnnSetConvolutionNdDescriptor_v3(
                                cudnnConvolutionDescriptor_t        convDesc,
                                int                                 arrayLength,             /* nbDims-2 size */
                                const int                           padA[],
                                const int                           filterStrideA[],
                                const int                           upscaleA[],
                                cudnnConvolutionMode_t              mode,
                                cudnnDataType_t                     dataType );   // convolution data type

/* cudnnGetConvolutionNdDescriptor_v3 is now mapped to cudnnGetConvolutionNdDescriptor
   and will be removed at the same time thancudnnGetConvolutionNdDescriptor_v2
   Use cudnnGetConvolutionNdDescriptor instead
 */
cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionNdDescriptor_v3(
                                const cudnnConvolutionDescriptor_t  convDesc,
                                int                                 arrayLengthRequested,
                                int                                *arrayLength,
                                int                                 padA[],
                                int                                 strideA[],
                                int                                 upscaleA[],
                                cudnnConvolutionMode_t             *mode,
                                cudnnDataType_t                    *dataType );  // convolution data type

/* Helper function to return the dimensions of the output tensor given a convolution descriptor */
cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionNdForwardOutputDim(
                                const cudnnConvolutionDescriptor_t  convDesc,
                                const cudnnTensorDescriptor_t       inputTensorDesc,
                                const cudnnFilterDescriptor_t       filterDesc,
                                int                                 nbDims,
                                int                                 tensorOuputDimA[] );

/* Destroy an instance of convolution descriptor */
cudnnStatus_t CUDNNWINAPI cudnnDestroyConvolutionDescriptor(
                                cudnnConvolutionDescriptor_t        convDesc );


/* helper function to provide the convolution algo that fit best the requirement */
typedef enum
{
    CUDNN_CONVOLUTION_FWD_NO_WORKSPACE            = 0,
    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST          = 1,
    CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT = 2,
} cudnnConvolutionFwdPreference_t;


typedef enum
{
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM         = 0,
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = 1,
    CUDNN_CONVOLUTION_FWD_ALGO_GEMM                  = 2,
    CUDNN_CONVOLUTION_FWD_ALGO_DIRECT                = 3,
    CUDNN_CONVOLUTION_FWD_ALGO_FFT                   = 4,
    CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING            = 5
} cudnnConvolutionFwdAlgo_t;

typedef struct {
    cudnnConvolutionFwdAlgo_t   algo;
    cudnnStatus_t               status;
    float                       time;
    size_t                      memory;
} cudnnConvolutionFwdAlgoPerf_t;

cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionForwardAlgorithm(
                                cudnnHandle_t                       handle,
                                const cudnnTensorDescriptor_t       xDesc,
                                const cudnnFilterDescriptor_t       wDesc,
                                const cudnnConvolutionDescriptor_t  convDesc,
                                const cudnnTensorDescriptor_t       yDesc,
                                const int                           requestedAlgoCount,
                                int                                *returnedAlgoCount,
                                cudnnConvolutionFwdAlgoPerf_t      *perfResults );

cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionForwardAlgorithm(
                                cudnnHandle_t                       handle,
                                const cudnnTensorDescriptor_t       xDesc,
                                const cudnnFilterDescriptor_t       wDesc,
                                const cudnnConvolutionDescriptor_t  convDesc,
                                const cudnnTensorDescriptor_t       yDesc,
                                cudnnConvolutionFwdPreference_t     preference,
                                size_t                              memoryLimitInBytes,
                                cudnnConvolutionFwdAlgo_t          *algo );

/*
 *  convolution algorithm (which requires potentially some workspace)
 */

 /* Helper function to return the minimum size of the workspace to be passed to the convolution given an algo*/
cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionForwardWorkspaceSize(
                                cudnnHandle_t                       handle,
                                const cudnnTensorDescriptor_t       xDesc,
                                const cudnnFilterDescriptor_t       wDesc,
                                const cudnnConvolutionDescriptor_t  convDesc,
                                const cudnnTensorDescriptor_t       yDesc,
                                cudnnConvolutionFwdAlgo_t           algo,
                                size_t                             *sizeInBytes );


/* Convolution functions: All of the form "output = alpha * Op(inputs) + beta * output" */

/* Function to perform the forward pass for batch convolution */
cudnnStatus_t CUDNNWINAPI cudnnConvolutionForward(
                                cudnnHandle_t                       handle,
                                const void                         *alpha,
                                const cudnnTensorDescriptor_t       xDesc,
                                const void                         *x,
                                const cudnnFilterDescriptor_t       wDesc,
                                const void                         *w,
                                const cudnnConvolutionDescriptor_t  convDesc,
                                cudnnConvolutionFwdAlgo_t           algo,
                                void                               *workSpace,
                                size_t                              workSpaceSizeInBytes,
                                const void                         *beta,
                                const cudnnTensorDescriptor_t       yDesc,
                                void                               *y );

/* Function to compute the bias gradient for batch convolution */
cudnnStatus_t CUDNNWINAPI cudnnConvolutionBackwardBias(
                                cudnnHandle_t                       handle,
                                const void                         *alpha,
                                const cudnnTensorDescriptor_t       dyDesc,
                                const void                         *dy,
                                const void                         *beta,
                                const cudnnTensorDescriptor_t       dbDesc,
                                void                               *db );


/* helper function to provide the convolution algo that fit best the requirement */
typedef enum
{
    CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE            = 0,
    CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST          = 1,
    CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT = 2,
} cudnnConvolutionBwdFilterPreference_t;

typedef enum
{
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0         = 0,  // non-deterministic
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1         = 1,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT       = 2,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3         = 3   // non-deterministic, algo0 with workspace
} cudnnConvolutionBwdFilterAlgo_t;


typedef struct {
    cudnnConvolutionBwdFilterAlgo_t algo;
    cudnnStatus_t status;
    float time;
    size_t memory;
} cudnnConvolutionBwdFilterAlgoPerf_t;

cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionBackwardFilterAlgorithm(
                                cudnnHandle_t                       handle,
                                const cudnnTensorDescriptor_t       xDesc,
                                const cudnnTensorDescriptor_t       dyDesc,
                                const cudnnConvolutionDescriptor_t  convDesc,
                                const cudnnFilterDescriptor_t       dwDesc,
                                const int                           requestedAlgoCount,
                                int                                 *returnedAlgoCount,
                                cudnnConvolutionBwdFilterAlgoPerf_t *perfResults );

cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardFilterAlgorithm(
                                cudnnHandle_t                         handle,
                                const cudnnTensorDescriptor_t         xDesc,
                                const cudnnTensorDescriptor_t         dyDesc,
                                const cudnnConvolutionDescriptor_t    convDesc,
                                const cudnnFilterDescriptor_t         dwDesc,
                                cudnnConvolutionBwdFilterPreference_t preference,
                                size_t                                memoryLimitInBytes,
                                cudnnConvolutionBwdFilterAlgo_t      *algo );

/*
 *  convolution algorithm (which requires potentially some workspace)
 */

 /* Helper function to return the minimum size of the workspace to be passed to the convolution given an algo*/
cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardFilterWorkspaceSize(
                                cudnnHandle_t                       handle,
                                const cudnnTensorDescriptor_t       xDesc,
                                const cudnnTensorDescriptor_t       dyDesc,
                                const cudnnConvolutionDescriptor_t  convDesc,
                                const cudnnFilterDescriptor_t       gradDesc,
                                cudnnConvolutionBwdFilterAlgo_t     algo,
                                size_t                             *sizeInBytes );

cudnnStatus_t CUDNNWINAPI cudnnConvolutionBackwardFilter(
                                cudnnHandle_t                       handle,
                                const void                         *alpha,
                                const cudnnTensorDescriptor_t       xDesc,
                                const void                         *x,
                                const cudnnTensorDescriptor_t       dyDesc,
                                const void                         *dy,
                                const cudnnConvolutionDescriptor_t  convDesc,
                                cudnnConvolutionBwdFilterAlgo_t     algo,
                                void                               *workSpace,
                                size_t                              workSpaceSizeInBytes,
                                const void                         *beta,
                                const cudnnFilterDescriptor_t       dwDesc,
                                void                               *dw );

/* cudnnConvolutionBackwardFilter_v3 is now mapped to cudnnConvolutionBackwardFilter
   and will be removed at the same time thancudnnConvolutionBackwardFilter_v2
   Use cudnnConvolutionBackwardFilter instead */
cudnnStatus_t CUDNNWINAPI cudnnConvolutionBackwardFilter_v3(
                                cudnnHandle_t                       handle,
                                const void                         *alpha,
                                const cudnnTensorDescriptor_t       xDesc,
                                const void                         *x,
                                const cudnnTensorDescriptor_t       dyDesc,
                                const void                         *dy,
                                const cudnnConvolutionDescriptor_t  convDesc,
                                cudnnConvolutionBwdFilterAlgo_t     algo,
                                void                               *workSpace,
                                size_t                              workSpaceSizeInBytes,
                                const void                         *beta,
                                const cudnnFilterDescriptor_t       dwDesc,
                                void                               *dw );
                                                           
/*********************************************************/
/* helper function to provide the convolution algo that fit best the requirement */
typedef enum
{
    CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE             = 0,
    CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST           = 1,
    CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT  = 2,
} cudnnConvolutionBwdDataPreference_t;

typedef enum
{
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_0          = 0, // non-deterministic
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_1          = 1,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT        = 2,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING = 3
} cudnnConvolutionBwdDataAlgo_t;

typedef struct {
    cudnnConvolutionBwdDataAlgo_t   algo;
    cudnnStatus_t                   status;
    float                           time;
    size_t                          memory;
} cudnnConvolutionBwdDataAlgoPerf_t;


cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionBackwardDataAlgorithm(
                                cudnnHandle_t                       handle,
                                const cudnnFilterDescriptor_t       wDesc,
                                const cudnnTensorDescriptor_t       dyDesc,
                                const cudnnConvolutionDescriptor_t  convDesc,
                                const cudnnTensorDescriptor_t       dxDesc,
                                const int                           requestedAlgoCount,
                                int                                *returnedAlgoCount,
                                cudnnConvolutionBwdDataAlgoPerf_t  *perfResults );

cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardDataAlgorithm(
                                cudnnHandle_t                       handle,
                                const cudnnFilterDescriptor_t       wDesc,
                                const cudnnTensorDescriptor_t       dyDesc,
                                const cudnnConvolutionDescriptor_t  convDesc,
                                const cudnnTensorDescriptor_t       dxDesc,
                                cudnnConvolutionBwdDataPreference_t preference,
                                size_t                              memoryLimitInBytes,
                                cudnnConvolutionBwdDataAlgo_t      *algo );

 /* Helper function to return the minimum size of the workspace to be passed to the convolution given an algo*/
cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardDataWorkspaceSize(
                                cudnnHandle_t                       handle,
                                const cudnnFilterDescriptor_t       wDesc,
                                const cudnnTensorDescriptor_t       dyDesc,
                                const cudnnConvolutionDescriptor_t  convDesc,
                                const cudnnTensorDescriptor_t       dxDesc,
                                cudnnConvolutionBwdDataAlgo_t       algo,
                                size_t                             *sizeInBytes );


cudnnStatus_t CUDNNWINAPI cudnnConvolutionBackwardData(
                                cudnnHandle_t                       handle,
                                const void                         *alpha,
                                const cudnnFilterDescriptor_t       wDesc,
                                const void                         *w,
                                const cudnnTensorDescriptor_t       dyDesc,
                                const void                         *dy,
                                const cudnnConvolutionDescriptor_t  convDesc,
                                cudnnConvolutionBwdDataAlgo_t       algo,
                                void                               *workSpace,
                                size_t                              workSpaceSizeInBytes,
                                const void                         *beta,
                                const cudnnTensorDescriptor_t       dxDesc,
                                void                               *dx );

/* cudnnConvolutionBackwardData_v3 is now mapped to cudnnConvolutionBackwardData
   and will be removed at the same time thancudnnConvolutionBackwardData_v2
   Use cudnnConvolutionBackwardData instead */
cudnnStatus_t CUDNNWINAPI cudnnConvolutionBackwardData_v3(
                                cudnnHandle_t                       handle,
                                const void                         *alpha,
                                const cudnnFilterDescriptor_t       wDesc,
                                const void                         *w,
                                const cudnnTensorDescriptor_t       dyDesc,
                                const void                         *dy,
                                const cudnnConvolutionDescriptor_t  convDesc,
                                cudnnConvolutionBwdDataAlgo_t       algo,
                                void                               *workSpace,
                                size_t                              workSpaceSizeInBytes,
                                const void                         *beta,
                                const cudnnTensorDescriptor_t       dxDesc,
                                void                               *dx );

cudnnStatus_t CUDNNWINAPI cudnnIm2Col(
                                cudnnHandle_t                       handle,
                                const cudnnTensorDescriptor_t       xDesc,
                                const void                         *x,
                                const cudnnFilterDescriptor_t       wDesc,
                                const cudnnConvolutionDescriptor_t  convDesc,
                                void                               *colBuffer );


/*
 *  softmax algorithm
 */
typedef enum
{
    CUDNN_SOFTMAX_FAST     = 0,         /* straightforward implementation */
    CUDNN_SOFTMAX_ACCURATE = 1,         /* subtract max from every point to avoid overflow */
    CUDNN_SOFTMAX_LOG      = 2
} cudnnSoftmaxAlgorithm_t;

typedef enum
{
    CUDNN_SOFTMAX_MODE_INSTANCE = 0,   /* compute the softmax over all C, H, W for each N */
    CUDNN_SOFTMAX_MODE_CHANNEL = 1     /* compute the softmax over all C for each H, W, N */
} cudnnSoftmaxMode_t;

/* Softmax functions: All of the form "output = alpha * Op(inputs) + beta * output" */

/* Function to perform forward softmax */
cudnnStatus_t CUDNNWINAPI cudnnSoftmaxForward(
                                cudnnHandle_t                       handle,
                                cudnnSoftmaxAlgorithm_t             algo,
                                cudnnSoftmaxMode_t                  mode,
                                const void                         *alpha,
                                const cudnnTensorDescriptor_t       xDesc,
                                const void                         *x,
                                const void                         *beta,
                                const cudnnTensorDescriptor_t       yDesc,
                                void                               *y );

/* Function to perform backward softmax */
cudnnStatus_t CUDNNWINAPI cudnnSoftmaxBackward(
                                cudnnHandle_t                       handle,
                                cudnnSoftmaxAlgorithm_t             algo,
                                cudnnSoftmaxMode_t                  mode,
                                const void                         *alpha,
                                const cudnnTensorDescriptor_t       yDesc,
                                const void                         *y,
                                const cudnnTensorDescriptor_t       dyDesc,
                                const void                         *dy,
                                const void                         *beta,
                                const cudnnTensorDescriptor_t       dxDesc,
                                void                               *dx );

/*
 *  pooling mode
 */
typedef enum
{
    CUDNN_POOLING_MAX     = 0,
    CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING = 1, // count for average includes padded values
    CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING = 2 // count for average does not include padded values
} cudnnPoolingMode_t;

/* Create an instance of pooling descriptor */
cudnnStatus_t CUDNNWINAPI cudnnCreatePoolingDescriptor(
                                cudnnPoolingDescriptor_t           *poolingDesc );

cudnnStatus_t CUDNNWINAPI cudnnSetPooling2dDescriptor(
                                cudnnPoolingDescriptor_t            poolingDesc,
                                cudnnPoolingMode_t                  mode,
                                int                                 windowHeight,
                                int                                 windowWidth,
                                int                                 verticalPadding,
                                int                                 horizontalPadding,
                                int                                 verticalStride,
                                int                                 horizontalStride );

cudnnStatus_t CUDNNWINAPI cudnnSetPooling2dDescriptor_v4(
                                cudnnPoolingDescriptor_t            poolingDesc,
                                cudnnPoolingMode_t                  mode,
                                cudnnNanPropagation_t               maxpoolingNanOpt,
                                int                                 windowHeight,
                                int                                 windowWidth,
                                int                                 verticalPadding,
                                int                                 horizontalPadding,
                                int                                 verticalStride,
                                int                                 horizontalStride );

cudnnStatus_t CUDNNWINAPI cudnnGetPooling2dDescriptor(
                                const cudnnPoolingDescriptor_t      poolingDesc,
                                cudnnPoolingMode_t                 *mode,
                                int                                *windowHeight,
                                int                                *windowWidth,
                                int                                *verticalPadding,
                                int                                *horizontalPadding,
                                int                                *verticalStride,
                                int                                *horizontalStride );

cudnnStatus_t CUDNNWINAPI cudnnGetPooling2dDescriptor_v4(
                                const cudnnPoolingDescriptor_t      poolingDesc,
                                cudnnPoolingMode_t                 *mode,
                                cudnnNanPropagation_t              *maxpoolingNanOpt,
                                int                                *windowHeight,
                                int                                *windowWidth,
                                int                                *verticalPadding,
                                int                                *horizontalPadding,
                                int                                *verticalStride,
                                int                                *horizontalStride );

cudnnStatus_t CUDNNWINAPI cudnnSetPoolingNdDescriptor(
                                cudnnPoolingDescriptor_t            poolingDesc,
                                const cudnnPoolingMode_t            mode,
                                int                                 nbDims,
                                const int                           windowDimA[],
                                const int                           paddingA[],
                                const int                           strideA[] );

cudnnStatus_t CUDNNWINAPI cudnnSetPoolingNdDescriptor_v4(
                                cudnnPoolingDescriptor_t            poolingDesc,
                                const cudnnPoolingMode_t            mode,
                                const cudnnNanPropagation_t         maxpoolingNanOpt,
                                int                                 nbDims,
                                const int                           windowDimA[],
                                const int                           paddingA[],
                                const int                           strideA[] );

cudnnStatus_t CUDNNWINAPI cudnnGetPoolingNdDescriptor(
                                const cudnnPoolingDescriptor_t      poolingDesc,
                                const int                           nbDimsRequested,
                                cudnnPoolingMode_t                 *mode,
                                int                                *nbDims,
                                int                                 windowDimA[],
                                int                                 paddingA[],
                                int                                 strideA[] );

cudnnStatus_t CUDNNWINAPI cudnnGetPoolingNdDescriptor_v4(
                                const cudnnPoolingDescriptor_t      poolingDesc,
                                int                                 nbDimsRequested,
                                cudnnPoolingMode_t                 *mode,
                                cudnnNanPropagation_t              *maxpoolingNanOpt,
                                int                                *nbDims,
                                int                                 windowDimA[],
                                int                                 paddingA[],
                                int                                 strideA[] );

cudnnStatus_t CUDNNWINAPI cudnnGetPoolingNdForwardOutputDim(
                                const cudnnPoolingDescriptor_t      poolingDesc,
                                const cudnnTensorDescriptor_t       inputTensorDesc,
                                int                                 nbDims,
                                int                                 outputTensorDimA[] );

cudnnStatus_t CUDNNWINAPI cudnnGetPooling2dForwardOutputDim(
                                const cudnnPoolingDescriptor_t      poolingDesc,
                                const cudnnTensorDescriptor_t       inputTensorDesc,
                                int                                *n,
                                int                                *c,
                                int                                *h,
                                int                                *w );


/* Destroy an instance of pooling descriptor */
cudnnStatus_t CUDNNWINAPI cudnnDestroyPoolingDescriptor(
                                cudnnPoolingDescriptor_t            poolingDesc );

/* Pooling functions: All of the form "output = alpha * Op(inputs) + beta * output" */

/* Function to perform forward pooling */
cudnnStatus_t CUDNNWINAPI cudnnPoolingForward(
                                cudnnHandle_t                       handle,
                                const cudnnPoolingDescriptor_t      poolingDesc,
                                const void                         *alpha,
                                const cudnnTensorDescriptor_t       xDesc,
                                const void                         *x,
                                const void                         *beta,
                                const cudnnTensorDescriptor_t       yDesc,
                                void                               *y );

/* Function to perform backward pooling */
cudnnStatus_t CUDNNWINAPI cudnnPoolingBackward(
                                cudnnHandle_t                       handle,
                                const cudnnPoolingDescriptor_t      poolingDesc,
                                const void                          *alpha,
                                const cudnnTensorDescriptor_t       yDesc,
                                const void                         *y,
                                const cudnnTensorDescriptor_t       dyDesc,
                                const void                         *dy,
                                const cudnnTensorDescriptor_t       xDesc,
                                const void                         *x,
                                const void                         *beta,
                                const cudnnTensorDescriptor_t       dxDesc,
                                void                               *dx );

/*
 * activation mode
 */
typedef enum
{
    CUDNN_ACTIVATION_SIGMOID      = 0,
    CUDNN_ACTIVATION_RELU         = 1,
    CUDNN_ACTIVATION_TANH         = 2,
    CUDNN_ACTIVATION_CLIPPED_RELU = 3
} cudnnActivationMode_t;

/* Activation functions: All of the form "output = alpha * Op(inputs) + beta * output" */
cudnnStatus_t CUDNNWINAPI cudnnCreateActivationDescriptor(
                                cudnnActivationDescriptor_t        *activationDesc);

cudnnStatus_t CUDNNWINAPI cudnnSetActivationDescriptor(
                                cudnnActivationDescriptor_t         activationDesc,
                                cudnnActivationMode_t               mode,
                                cudnnNanPropagation_t               reluNanOpt,
                                double                              reluCeiling );

cudnnStatus_t CUDNNWINAPI cudnnGetActivationDescriptor(
                                const cudnnActivationDescriptor_t   activationDesc,
                                cudnnActivationMode_t              *mode,
                                cudnnNanPropagation_t              *reluNanOpt,
                                double*                             reluCeiling );

cudnnStatus_t CUDNNWINAPI cudnnDestroyActivationDescriptor(
                                cudnnActivationDescriptor_t activationDesc);

/* Function to perform forward activation  */
cudnnStatus_t CUDNNWINAPI cudnnActivationForward(
                                cudnnHandle_t                       handle,
                                cudnnActivationMode_t               mode,
                                const void                         *alpha,
                                const cudnnTensorDescriptor_t       xDesc,
                                const void                         *x,
                                const void                         *beta,
                                const cudnnTensorDescriptor_t       yDesc,
                                void                               *y );

cudnnStatus_t CUDNNWINAPI cudnnActivationForward_v4(
                                cudnnHandle_t                       handle,
                                cudnnActivationDescriptor_t         activationDesc,
                                const void                         *alpha,
                                const cudnnTensorDescriptor_t       xDesc,
                                const void                         *x,
                                const void                         *beta,
                                const cudnnTensorDescriptor_t       yDesc,
                                void                               *y );

/* Function to perform backward activation  */
cudnnStatus_t CUDNNWINAPI cudnnActivationBackward(
                                cudnnHandle_t                       handle,
                                cudnnActivationMode_t               mode,
                                const void                         *alpha,
                                const cudnnTensorDescriptor_t       yDesc,
                                const void                         *y,
                                const cudnnTensorDescriptor_t       dyDesc,
                                const void                         *dy,
                                const cudnnTensorDescriptor_t       xDesc,
                                const void                         *x,
                                const void                         *beta,
                                const cudnnTensorDescriptor_t       dxDesc,
                                void                               *dx );

cudnnStatus_t CUDNNWINAPI cudnnActivationBackward_v4(
                                cudnnHandle_t                       handle,
                                cudnnActivationDescriptor_t         activationDesc,
                                const void                         *alpha,
                                const cudnnTensorDescriptor_t       yDesc,
                                const void                         *y,
                                const cudnnTensorDescriptor_t       dyDesc,
                                const void                         *dy,
                                const cudnnTensorDescriptor_t       xDesc,
                                const void                         *x,
                                const void                         *beta,
                                const cudnnTensorDescriptor_t       dxDesc,
                                void                               *dx );

/* 
* Create an instance of LRN (Local Response Normalization) descriptor
* Uses lrnN=5, lrnAlpha=1e-4, lrnBeta=0.75, lrnK=2.0 as defaults from Krizhevsky'12 ImageNet paper
*/
cudnnStatus_t CUDNNWINAPI cudnnCreateLRNDescriptor(
                                cudnnLRNDescriptor_t               *normDesc );

#define CUDNN_LRN_MIN_N     1       // minimum allowed lrnN
#define CUDNN_LRN_MAX_N     16      // maximum allowed lrnN
#define CUDNN_LRN_MIN_K     1e-5    // minimum allowed lrnK
#define CUDNN_LRN_MIN_BETA  0.01    // minimum allowed lrnBeta

/* LRN layer mode */
typedef enum
{
    CUDNN_LRN_CROSS_CHANNEL_DIM1 = 0,// Normalize across tensor's dimA[1] dimension
} cudnnLRNMode_t;

/*
* Uses a window [center-lookBehind, center+lookAhead], where
* lookBehind = floor( (lrnN-1)/2 ), lookAhead = lrnN-lookBehind-1.
* Values of double parameters cast to tensor data type.
*/
cudnnStatus_t CUDNNWINAPI cudnnSetLRNDescriptor(
                                cudnnLRNDescriptor_t                normDesc,
                                unsigned                            lrnN,
                                double                              lrnAlpha,
                                double                              lrnBeta,
                                double                              lrnK );
/*
* Retrieve the settings currently stored in an LRN layer descriptor
* Any of the provided pointers can be NULL (no corresponding value will be returned)
*/
cudnnStatus_t CUDNNWINAPI cudnnGetLRNDescriptor(
                                cudnnLRNDescriptor_t                normDesc,
                                unsigned*                           lrnN,
                                double*                             lrnAlpha,
                                double*                             lrnBeta,
                                double*                             lrnK );

/* Destroy an instance of LRN descriptor */
cudnnStatus_t CUDNNWINAPI cudnnDestroyLRNDescriptor( cudnnLRNDescriptor_t lrnDesc );

/* LRN functions: output = alpha * normalize(x) + beta * old_y */

/* LRN cross-channel forward computation. Double parameters cast to tensor data type */
cudnnStatus_t CUDNNWINAPI cudnnLRNCrossChannelForward(
                                cudnnHandle_t                       handle,
                                cudnnLRNDescriptor_t                normDesc,
                                cudnnLRNMode_t                      lrnMode,
                                const void*                         alpha,
                                const cudnnTensorDescriptor_t       xDesc,
                                const void                         *x,
                                const void                         *beta,
                                const cudnnTensorDescriptor_t       yDesc,
                                void                               *y );

/* LRN cross-channel backward computation. Double parameters cast to tensor data type */
cudnnStatus_t CUDNNWINAPI cudnnLRNCrossChannelBackward(
                                cudnnHandle_t                       handle,
                                cudnnLRNDescriptor_t                normDesc,
                                cudnnLRNMode_t                      lrnMode,
                                const void*                         alpha,
                                const cudnnTensorDescriptor_t       yDesc,
                                const void                         *y,
                                const cudnnTensorDescriptor_t       dyDesc,
                                const void                         *dy,
                                const cudnnTensorDescriptor_t       xDesc,
                                const void                         *x,
                                const void                         *beta,
                                const cudnnTensorDescriptor_t       dxDesc,
                                void                               *dx);

typedef enum
{
    CUDNN_DIVNORM_PRECOMPUTED_MEANS = 0,
} cudnnDivNormMode_t;

/* LCN/divisive normalization functions: y = alpha * normalize(x) + beta * y */
cudnnStatus_t CUDNNWINAPI cudnnDivisiveNormalizationForward(
                                cudnnHandle_t                       handle,
                                cudnnLRNDescriptor_t                normDesc,
                                cudnnDivNormMode_t                  mode,
                                const void                         *alpha,
                                const cudnnTensorDescriptor_t       xDesc, // same desc for means, temp, temp2
                                const void                         *x,
                                const void                         *means, // if NULL, means are assumed to be zero
                                void                               *temp,
                                void                               *temp2,
                                const void                         *beta,
                                const cudnnTensorDescriptor_t       yDesc,
                                void                               *y );

cudnnStatus_t CUDNNWINAPI cudnnDivisiveNormalizationBackward(
                                cudnnHandle_t                       handle,
                                cudnnLRNDescriptor_t                normDesc,
                                cudnnDivNormMode_t                  mode,
                                const void                         *alpha,
                                const cudnnTensorDescriptor_t       xDesc, // same desc for x, means, dy, temp, temp2
                                const void                         *x,
                                const void                         *means, // if NULL, means are assumed to be zero
                                const void                         *dy,
                                void                               *temp,
                                void                               *temp2,
                                const void                         *beta,
                                const cudnnTensorDescriptor_t       dXdMeansDesc, // same desc for dx, dMeans
                                void                               *dx, // output x differential
                                void                               *dMeans ); // output means differential, can be NULL

typedef enum
{
    // bnScale, bnBias tensor dims are 1xCxHxWx.. (one value per CHW...-slice, normalized over N slice)
    CUDNN_BATCHNORM_PER_ACTIVATION = 0,

    //bnScale, bnBias tensor dims are 1xCx1x1 (one value per C-dim normalized over Nx1xHxW subtensors)
    CUDNN_BATCHNORM_SPATIAL        = 1,
} cudnnBatchNormMode_t;

#define CUDNN_BN_MIN_EPSILON 1e-5 // Minimum epsilon allowed to be used in the Batch Normalization formula

/*
* Derives a tensor descriptor from layer data descriptor for BatchNormalization 
* scale, invVariance, bnBias, bnScale tensors. Use this tensor desc for 
* bnScaleBiasMeanVarDesc and bnScaleBiasDiffDesc in Batch Normalization forward and backward functions.
*/
cudnnStatus_t CUDNNWINAPI cudnnDeriveBNTensorDescriptor(
                                cudnnTensorDescriptor_t             derivedBnDesc,
                                const cudnnTensorDescriptor_t       xDesc,
                                cudnnBatchNormMode_t                mode );

/* Computes y = BN(x). Also accumulates moving averages of mean and inverse variances */
cudnnStatus_t CUDNNWINAPI cudnnBatchNormalizationForwardTraining(
                                cudnnHandle_t                       handle,
                                cudnnBatchNormMode_t                mode,

                                const void                         *alpha, // alpha[0] = result blend factor
                                const void                         *beta,  // beta[0] = dest layer blend factor

                                const cudnnTensorDescriptor_t       xDesc,
                                const void                         *x,     // NxCxHxW
                                const cudnnTensorDescriptor_t       yDesc,
                                void                               *y,     // NxCxHxW

                                /* Shared desc for the next 6 tensors in the argument list.
                                   Data type to be set as follows:
                                   type = (typeOf(x) == double) ? double : float
                                   Dimensions for this descriptor depend on normalization mode
                                   - Spatial Normalization : tensors are expected to have dims 1xCx1x1
                                    (normalization is performed across NxHxW)
                                   - Per-Activation Normalization : tensors are expected to have dims of 1xCxHxW 
                                    (normalization is performed across N) */
                                const cudnnTensorDescriptor_t       bnScaleBiasMeanVarDesc,

                                // 'Gamma' and 'Beta' respectively in Ioffe and Szegedy's paper's notation
                                const void                         *bnScale,
                                const void                         *bnBias,

                                /* MUST use factor=1 in the very first call of a complete training cycle.
                                   Use a factor=1/(1+n) at N-th call to the function to get
                                   Cumulative Moving Average (CMA) behavior
                                   CMA[n] = (x[1]+...+x[n])/n
                                   Since CMA[n+1] = (n*CMA[n]+x[n+1])/(n+1) =
                                   ((n+1)*CMA[n]-CMA[n])/(n+1) + x[n+1]/(n+1) =
                                   CMA[n]*(1-1/(n+1)) + x[n+1]*1/(n+1) */
                                double                              exponentialAverageFactor,

                                /* Used in Training phase only. 
                                   runningMean = newMean*factor + runningMean*(1-factor) */
                                void                               *resultRunningMean,
                                /* Output in training mode, input in inference. Is the moving average
                                   of 1 / sqrt( epsilon + variance[x] ) */
                                void                               *resultRunningInvVariance,

                                /* Has to be >= CUDNN_BN_MIN_EPSILON. Should be the same in forward and backward functions. */
                                double                              epsilon,

                                /* Optionally save intermediate results from the forward pass here
                                   - can be reused to speed up backward pass. NULL if unused */
                                void                               *resultSaveMean,
                                void                               *resultSaveInvVariance );

/*
* Performs Batch Normalization during Inference: 
* y[i] = bnScale[k]*(x[i]-estimatedMean[k])*estimatedInvVariance[k] + bnBias[k]
* with bnScale, bnBias, runningMean, runningInvVariance tensors indexed
* according to spatial or per-activation mode. Refer to cudnnBatchNormalizationForwardTraining
* above for notes on function arguments.
*/
cudnnStatus_t CUDNNWINAPI cudnnBatchNormalizationForwardInference(
                                cudnnHandle_t                       handle,
                                cudnnBatchNormMode_t                mode,
                                const void                         *alpha, // alpha[0] = result blend factor
                                const void                         *beta,  // beta[0] = dest layer blend factor
                                const cudnnTensorDescriptor_t       xDesc,
                                const void                         *x,     // NxCxHxW
                                const cudnnTensorDescriptor_t       yDesc,
                                void                               *y,     // NxCxHxW
                                const cudnnTensorDescriptor_t       bnScaleBiasMeanVarDesc,
                                const void                         *bnScale,
                                const void                         *bnBias,
                                const void                         *estimatedMean,
                                const void                         *estimatedInvVariance,
                                double                              epsilon );

/* Performs backward pass of Batch Normalization layer. Returns x gradient,
* bnScale gradient and bnBias gradient */
cudnnStatus_t CUDNNWINAPI cudnnBatchNormalizationBackward(
                                cudnnHandle_t                       handle,
                                cudnnBatchNormMode_t                mode,
                                const void                         *alpha,
                                const void                         *beta,
                                const cudnnTensorDescriptor_t       xDesc, // same desc for x, dx, dy
                                const void                         *x,
                                const cudnnTensorDescriptor_t       dyDesc,
                                const void                         *dy,
                                const cudnnTensorDescriptor_t       dxDesc,
                                void                               *dx,
                                /* Shared tensor desc for the 4 tensors below */
                                const cudnnTensorDescriptor_t       dBnScaleBiasDesc,
                                const void                         *bnScale, // bnBias doesn't affect backpropagation
                                /* scale and bias diff are not backpropagated below this layer */
                                void                               *dBnScaleResult,
                                void                               *dBnBiasResult,
                                /* Same epsilon as forward pass */
                                double                              epsilon,

                                /* Optionally cached intermediate results from
                                   forward pass */
                                const void                         *savedMean,
                                const void                         *savedInvVariance );

/* DEPRECATED API THAT WILL BE REMOVED FROM UPCOMING RELEASES */
cudnnStatus_t CUDNNWINAPI cudnnSetConvolutionNdDescriptor_v2(
                                cudnnConvolutionDescriptor_t convDesc,
                                int arrayLength,             /* nbDims-2 size */
                                const int padA[],
                                const int filterStrideA[],
                                const int upscaleA[],
                                cudnnConvolutionMode_t mode );

cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionNdDescriptor_v2(
                                const cudnnConvolutionDescriptor_t  convDesc,
                                int                                 arrayLengthRequested,
                                int                                *arrayLength,
                                int                                 padA[],
                                int                                 strideA[],
                                int                                 upscaleA[],
                                cudnnConvolutionMode_t             *mode );
                                                         
cudnnStatus_t CUDNNWINAPI cudnnAddTensor_v2(
                                cudnnHandle_t                       handle,
                                cudnnAddMode_t                      mode,
                                const void                         *alpha,
                                const cudnnTensorDescriptor_t       bDesc,
                                const void                         *b,
                                const void                         *beta,
                                cudnnTensorDescriptor_t             yDesc,
                                void                               *y );
                                            
cudnnStatus_t CUDNNWINAPI cudnnConvolutionBackwardFilter_v2(
                                cudnnHandle_t                       handle,
                                const void                         *alpha,
                                const cudnnTensorDescriptor_t       xDesc,
                                const void                         *x,
                                const cudnnTensorDescriptor_t       dyDesc,
                                const void                         *dy,
                                const cudnnConvolutionDescriptor_t  convDesc,
                                const void                         *beta,
                                const cudnnFilterDescriptor_t       dxDesc,
                                void                               *dx );
                                                           
cudnnStatus_t CUDNNWINAPI cudnnConvolutionBackwardData_v2(
                                cudnnHandle_t                       handle,
                                const void                         *alpha,
                                const cudnnFilterDescriptor_t       xDesc,
                                const void                         *x,
                                const cudnnTensorDescriptor_t       dyDesc,
                                const void                         *dy,
                                const cudnnConvolutionDescriptor_t  convDesc,
                                const void                         *beta,
                                const cudnnTensorDescriptor_t       dxDesc,
                                void                               *dx );

#if defined (__cplusplus)
}
#endif

#endif /* CUDNN_H_ */
