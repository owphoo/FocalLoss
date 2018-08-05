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
void FocalLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  has_ignore_label_ = this->layer_param_.loss_param().has_ignore_label();

  if (has_ignore_label_)
  {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  if (!this->layer_param_.loss_param().has_normalization() && this->layer_param_.loss_param().has_normalize())
  {
    normalization_ = this->layer_param_.loss_param().normalize() ? LossParameter_NormalizationMode_VALID : LossParameter_NormalizationMode_BATCH_SIZE;
  }
  else
  {
    normalization_ = this->layer_param_.loss_param().normalization();
    // printf("noralization: %d\n", (int)normalization_);
  }
  gamma_ = this->layer_param_.focal_loss_param().gamma();
  alpha_balance_ = false;
  if (this->layer_param_.focal_loss_param().has_alpha_balance())
  { 
    alpha_balance_ = this->layer_param_.focal_loss_param().alpha_balance();
  }

}

template <typename Dtype>
void FocalLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  softmax_axis_ = bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";

  alpha_.Reshape(1, 1, 1, bottom[0]->shape(softmax_axis_));
  Dtype* alpha_data = alpha_.mutable_cpu_data();
  if(alpha_balance_)
  {
    int sz = this->layer_param_.focal_loss_param().alpha_size();
    //alpha's size must equal to the class number
    CHECK_EQ(sz, bottom[0]->shape(softmax_axis_));

    for(int i = 0; i < sz; ++ i)
    {
      alpha_data[i] = this->layer_param_.focal_loss_param().alpha(i);
    }

  }else
  {
    for(int i = 0; i < alpha_.count(); ++ i)
    {
      alpha_data[i] = (Dtype)1;
    }

  }
  
  
  
  pow_prob_gamma_.ReshapeLike(prob_);
  
  if (top.size() >= 2)
  {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
Dtype FocalLossLayer<Dtype>::get_normalizer(LossParameter_NormalizationMode normalization_mode, int valid_count)
{
  Dtype normalizer;
  switch (normalization_mode)
  {
    case LossParameter_NormalizationMode_FULL:
      normalizer = Dtype(outer_num_ * inner_num_);
      break;
    case LossParameter_NormalizationMode_VALID:
      if (valid_count == -1)
      {
        normalizer = Dtype(outer_num_ * inner_num_);
      }
      else
      {
        normalizer = Dtype(valid_count);
      }
      break;
    case LossParameter_NormalizationMode_BATCH_SIZE:
      normalizer = Dtype(outer_num_);
      break;
    case LossParameter_NormalizationMode_NONE:
      normalizer = Dtype(1);
      break;
    default:
      LOG(FATAL) << "Unknown normalization mode: " << LossParameter_NormalizationMode_Name(normalization_mode);
  }
  // Some users will have no labels for some examples in order to 'turn off' a
  // particular loss in a multi-task setup. The max prevents NaNs in that case.
  return std::max(Dtype(1.0), normalizer);
}

template <typename Dtype>
void FocalLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);

  // printf("prob: %f %f\n", prob_.cpu_data()[0], prob_.cpu_data()[1]);
  //-prob
  caffe_copy(prob_.count(), prob_.cpu_data(), pow_prob_gamma_.mutable_cpu_data());
  caffe_scal(prob_.count(), Dtype(-1), pow_prob_gamma_.mutable_cpu_data());
  //1-prob to avoid 1-prob -> 0
  caffe_add_scalar(prob_.count(), Dtype(1.0+1e-6), pow_prob_gamma_.mutable_cpu_data());

  //(1-prob)^(gamma-1) for backward 
  caffe_powx(prob_.count(), pow_prob_gamma_.cpu_data(), gamma_-1, pow_prob_gamma_.mutable_cpu_diff());
  //(1-prob)^gamma
  caffe_powx(prob_.count(), pow_prob_gamma_.cpu_data(), gamma_, pow_prob_gamma_.mutable_cpu_data());
  
  

  // printf("pow_prob_gamma_: %f %f\n", pow_prob_gamma_.cpu_data()[0], pow_prob_gamma_.cpu_data()[1]);

 
  const Dtype* prob_data = prob_.cpu_data();
  // to save(pk-pl*pk-gamma*pk*log(pl))
  Dtype* prob_diff = prob_.mutable_cpu_diff();
  // caffe_copy(prob_.count(), prob_data, prob_diff);

  const Dtype* pow_prob_gamma_data = pow_prob_gamma_.cpu_data();
  const Dtype* pow_prob_gamma_diff = pow_prob_gamma_.cpu_diff();
  const Dtype* label = bottom[1]->cpu_data();
  const Dtype* alpha_data = alpha_.cpu_data();

  int dim = prob_.count() / outer_num_;

  int count = 0;
  Dtype loss = 0;


  for (int i = 0; i < outer_num_; ++i)
  {
    for (int j = 0; j < inner_num_; j++)
    {
      const int label_value = static_cast<int>(label[i * inner_num_ + j]);

      if (has_ignore_label_ && label_value == ignore_label_)
        continue;
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, prob_.shape(softmax_axis_));

      // Dtype tmp = prob_data[i * dim + label_value * inner_num_ + j];
      //loss = -alpha_c*(1-prob)^gamma*log(prob)
      Dtype p_l = prob_data[i * dim + label_value * inner_num_ + j];
      int label_idx = i * dim + label_value * inner_num_ + j;
      for(int k = 0; k < prob_.shape(softmax_axis_); ++ k)
      {
        if(k == label_value)
          continue;
        int k_idx = i * dim + k * inner_num_ + j;
        Dtype p_k = prob_data[k_idx];
        prob_diff[k_idx] = alpha_data[label_value] * pow_prob_gamma_diff[label_idx] * 
          (p_k - p_l * p_k * (1 + gamma_ * log( std::max(p_l, Dtype(FLT_MIN)))));

        // printf("k_idx: %f %f %f %d %f\n", pow_prob_gamma_diff[label_idx], p_k, p_l, k_idx, prob_diff[k_idx]);

      }

      loss -= (alpha_data[label_value]*(pow_prob_gamma_data[label_idx] * 
        log( std::max(p_l, Dtype(FLT_MIN)))));
      ++count;
    }
  }
  // printf("alpha:%f powgamma:%f gamma:%f prob:%f \n", alpha_data[0], pow_prob_gamma_data[0], gamma_,
  //      prob_data[0]);

  top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_, count);
  if (top.size() == 2)
  {
    top[1]->ShareData(prob_);
  }
}

/*
Backward
dl/dx = y(1-prob)^gamma(gamma * prob * log(prob) + prob - 1)

*/

template <typename Dtype>
void FocalLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
  if (propagate_down[1])
  {
    LOG(FATAL)<< this->type() << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0])
  {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    caffe_copy(prob_.count(), prob_.cpu_diff(), bottom_diff);
    // caffe_set(bottom[0]->count(), Dtype(0.), bottom_diff);
    const Dtype* label = bottom[1]->cpu_data();
    const Dtype* pow_prob_gamma_data = pow_prob_gamma_.cpu_data();


    const Dtype* alpha_data = alpha_.cpu_data();
    int dim = prob_.count() / outer_num_;
    int count = 0;
    for (int i = 0; i < outer_num_; ++i)
    {
      for (int j = 0; j < inner_num_; ++j)
      {
        const int label_value = static_cast<int>(label[i * inner_num_ + j]);
        if (has_ignore_label_ && label_value == ignore_label_)
        {
          for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c)
          {
            bottom_diff[i * dim + c * inner_num_ + j] = 0;
          }
        }
        else
        {
          // bottom_diff[i * dim + label_value * inner_num_ + j] -= 1;
          int label_idx = i * dim + label_value * inner_num_ + j;
          
         
          // bottom_diff[index] = alpha_data[label_value] * pow_prob_gamma_data[index] * (gamma_ * bottom_diff[index] * log(std::max(bottom_diff[index], Dtype(FLT_MIN))) + 
          //   bottom_diff[index] - 1);

          bottom_diff[label_idx] = alpha_data[label_value] * pow_prob_gamma_data[label_idx] * (gamma_ * prob_data[label_idx] * log(std::max(prob_data[label_idx], Dtype(FLT_MIN))) + 
            prob_data[label_idx] - 1);
          ++count;
        }
      }
    }
    // Scale gradient

    Dtype loss_weight = top[0]->cpu_diff()[0] / get_normalizer(normalization_, count);
    caffe_scal(prob_.count(), loss_weight, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(FocalLossLayer);
#endif

INSTANTIATE_CLASS(FocalLossLayer);
REGISTER_LAYER_CLASS(FocalLoss);

}  // namespace caffe
