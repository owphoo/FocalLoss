/*
Author: syhoo
E-mail: sheyeehoo@gmail.com
*/

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/focal_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe 
{

template <typename Dtype>
__global__ void FocalLossForwardGPU(const int nthreads, const Dtype* prob_data,  
  const Dtype* pow_prob_gamma_data, const Dtype* pow_prob_gamma_diff, const Dtype* alpha_data, 
  const Dtype* label, Dtype* loss, const int num, const int dim, const int spatial_dim, 
  const bool has_ignore_label_, const int ignore_label_, const int channel, const Dtype gamma, 
  Dtype* counts, Dtype* prob_diff) 
{
  CUDA_KERNEL_LOOP(index, nthreads) 
  {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);
    if (has_ignore_label_ && label_value == ignore_label_) 
    {
      loss[index] = 0;
      counts[index] = 0;
    } 
    else 
    {

      int label_idx = n * dim + label_value * spatial_dim + s;
      Dtype p_l = prob_data[label_idx];
      // For Backward
      for(int k = 0; k < channel; ++ k)
      {
        if(k == label_value)
          continue;
        int k_idx = n * dim + k * spatial_dim + s;
        Dtype p_k = prob_data[k_idx];
        prob_diff[k_idx] = alpha_data[label_value] * pow_prob_gamma_diff[label_idx] * 
          (p_k - p_l * p_k * (1 + gamma * log( max(p_l, Dtype(FLT_MIN)))));

        // printf("k_idx: %f %f %f %d %f\n", pow_prob_gamma_diff[label_idx], p_k, p_l, k_idx, prob_diff[k_idx]);

      }

      loss[index] = -alpha_data[label_value] * log(max(p_l, Dtype(FLT_MIN))) * pow_prob_gamma_data[label_idx];
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
void FocalLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  // Forward_cpu(bottom, top);
  //      return ;
///*    
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);

  //-prob
  caffe_copy(prob_.count(), prob_.cpu_data(), pow_prob_gamma_.mutable_cpu_data());
  // caffe_gpu_memcpy(prob_.count() * sizeof(Dtype), prob_.gpu_data(), pow_prob_gamma_.mutable_gpu_data());
  caffe_scal(prob_.count(), Dtype(-1), pow_prob_gamma_.mutable_cpu_data());
  //1-prob to avoid 1-prob -> 0
  caffe_add_scalar(prob_.count(), Dtype(1.0+1e-6), pow_prob_gamma_.mutable_cpu_data());

  //(1-prob)^(gamma-1) for backward 
  caffe_powx(prob_.count(), pow_prob_gamma_.cpu_data(), gamma_-1, pow_prob_gamma_.mutable_cpu_diff());
  //(1-prob)^gamma
  caffe_powx(prob_.count(), pow_prob_gamma_.cpu_data(), gamma_, pow_prob_gamma_.mutable_cpu_data());
  
  const Dtype* prob_data = prob_.gpu_data();
  const Dtype* pow_prob_gamma_data = pow_prob_gamma_.gpu_data();
  const Dtype* pow_prob_gamma_diff = pow_prob_gamma_.gpu_diff();
  const Dtype* alpha_data = alpha_.gpu_data();
  const Dtype* label = bottom[1]->gpu_data();
  const int dim = prob_.count() / outer_num_;
  const int nthreads = outer_num_ * inner_num_;
    
  // Since this memory is not used for anything until it is overwritten
  // on the backward pass, we use it here to avoid having to allocate new GPU
  // memory to accumulate intermediate results in the kernel.
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();
  
  Dtype* prob_diff = prob_.mutable_gpu_diff();
  

  //to save counts
  Blob<Dtype> blob_counts;
  blob_counts.ReshapeLike(prob_);
  caffe_set(prob_.count(), Dtype(0.0), blob_counts.mutable_cpu_data());
  Dtype* counts = blob_counts.mutable_gpu_data();
// FocalLossForwardGPU(const int nthreads, const Dtype* prob_data,  
//   const Dtype* pow_prob_gamma_data, const Dtype* pow_prob_gamma_diff, const Dtype* alpha_data, 
//   const Dtype* label, Dtype* loss, const int num, const int dim, const int spatial_dim, 
//   const bool has_ignore_label_, const int ignore_label_, const int channel, const Dtype gamma, 
//   Dtype* counts, Dtype* prob_diff) 


    
  // NOLINT_NEXT_LINE(whitespace/operators)
  FocalLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>
    (nthreads, prob_data, pow_prob_gamma_data, pow_prob_gamma_diff, alpha_data, label, loss_data, 
    outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, bottom[0]->shape(softmax_axis_),
    gamma_, counts, prob_diff);
  Dtype loss;
  caffe_gpu_asum(nthreads, loss_data, &loss);
  Dtype valid_count = -1;
    
  // Only launch another CUDA kernel if we actually need the count of valid outputs.
  if (normalization_ == LossParameter_NormalizationMode_VALID && has_ignore_label_) 
  {
    caffe_gpu_asum(nthreads, counts, &valid_count);
  }
  top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_,
                                                          valid_count);
  if (top.size() == 2) 
  {
    top[1]->ShareData(prob_);
  }
//*/

}

template <typename Dtype>
__global__ void FocalLossBackwardGPU(const int nthreads, const Dtype* top, const Dtype* prob_data, const Dtype* pow_prob_gamma_data, const Dtype* alpha_data, const Dtype* label, Dtype* bottom_diff, const Dtype gamma, const int num, const int dim, const int spatial_dim, const bool has_ignore_label_, const int ignore_label_, Dtype* counts) 
{

  const int channels = dim / spatial_dim;

  CUDA_KERNEL_LOOP(index, nthreads) 
  {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);

    if (has_ignore_label_ && label_value == ignore_label_) 
    {
      for (int c = 0; c < channels; ++c) 
      {
        bottom_diff[n * dim + c * spatial_dim + s] = 0;
      }
      counts[index] = 0;
    } 
    else 
    {
     // bottom_diff[n * dim + label_value * spatial_dim + s] -= 1;
       int idx = n * dim + label_value * spatial_dim + s;
       Dtype temp = prob_data[idx] < Dtype(FLT_MIN) ? Dtype(FLT_MIN) : prob_data[idx];
       
       bottom_diff[idx] = alpha_data[label_value] * pow_prob_gamma_data[idx] * (gamma * prob_data[idx] * log(temp) + 
            prob_data[idx] - Dtype(1.));
       // bottom_diff[idx] = alpha_data[label_value] * pow_prob_gamma_data[idx] * (gamma * bottom_diff[idx] * log(temp) + 
       //      bottom_diff[idx] - Dtype(1.));
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
void FocalLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
  // Backward_cpu(top, propagate_down, bottom);
  //      return;

///*
    if (propagate_down[1]) 
    {
      LOG(FATAL) << this->type() << " Layer cannot backpropagate to label inputs.";
    }
    if (propagate_down[0]) 
    {
      // caffe_set(bottom[0]->count(), Dtype(0.), bottom[0]->mutable_cpu_diff());
      
      Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
      const Dtype* prob_data = prob_.cpu_data();
      const Dtype* top_data = top[0]->gpu_data();
      const Dtype* pow_prob_gamma_data = pow_prob_gamma_.gpu_data();
      const Dtype* alpha_data = alpha_.gpu_data();

      // caffe_gpu_memcpy(prob_.count() * sizeof(Dtype), prob_data, bottom_diff);
      caffe_gpu_memcpy(prob_.count() * sizeof(Dtype), prob_.cpu_diff(), bottom_diff);
      const Dtype* label = bottom[1]->gpu_data();
      const int dim = prob_.count() / outer_num_;
      const int nthreads = outer_num_ * inner_num_;
    
      // Since this memory is never used for anything else,
      // we use to to avoid allocating new GPU memory.
      Blob<Dtype> blob_counts;
      blob_counts.ReshapeLike(prob_);
      // caffe_set(prob_.count(), Dtype(0.0), prob_.mutable_cpu_data());
      Dtype* counts = blob_counts.mutable_gpu_data();
      // Dtype* counts = prob_.mutable_gpu_diff();
    
      // NOLINT_NEXT_LINE(whitespace/operators)
      FocalLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(nthreads, top_data, prob_data, pow_prob_gamma_data, 
        alpha_data, label, bottom_diff, gamma_, outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, counts);

      Dtype valid_count = -1;
      // Only launch another CUDA kernel if we actually need the count of valid outputs.
      if (normalization_ == LossParameter_NormalizationMode_VALID && has_ignore_label_) 
      {
          caffe_gpu_asum(nthreads, counts, &valid_count);
      }
      const Dtype loss_weight = top[0]->cpu_diff()[0] / get_normalizer(normalization_, valid_count);
      caffe_gpu_scal(prob_.count(), loss_weight , bottom_diff);
    }
//*/
}

INSTANTIATE_LAYER_GPU_FUNCS(FocalLossLayer);

}  // namespace caffe
