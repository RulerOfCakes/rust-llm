use burn::{
    module::Module,
    nn::{Dropout, DropoutConfig, Linear, LinearConfig},
    prelude::Backend,
    tensor::{Shape, Tensor, activation::softmax},
};

pub struct CustomAttentionConfig {
    input_dim: usize,
    output_dim: usize,
}

impl CustomAttentionConfig {
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        Self {
            input_dim,
            output_dim,
        }
    }
    pub fn init<B: Backend>(&self, device: &B::Device, dropout: f64) -> CustomAttention<B> {
        let w_q = LinearConfig::new(self.input_dim, self.output_dim)
            .with_bias(false)
            .init(device);
        let w_k = LinearConfig::new(self.input_dim, self.output_dim)
            .with_bias(false)
            .init(device);
        let w_v = LinearConfig::new(self.input_dim, self.output_dim)
            .with_bias(false)
            .init(device);
        let dropout = DropoutConfig::new(dropout).init();
        CustomAttention {
            dk_norm_factor: (self.output_dim as f32).sqrt(),
            w_q,
            w_k,
            w_v,
            dropout,
        }
    }
}

#[derive(Debug, Module)]
pub struct CustomAttention<B: Backend> {
    dk_norm_factor: f32,
    w_q: Linear<B>, // [input_embed_dim, output_embed_dim]
    w_k: Linear<B>,
    w_v: Linear<B>,

    dropout: Dropout,
}

impl<B: Backend> CustomAttention<B> {
    // perform scaled dot-product attention (single head)
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let q = self.w_q.forward(input.clone()); // [batch_size, seq_len, output_embed_dim]
        let k = self.w_k.forward(input.clone()); // [batch_size, seq_len, output_embed_dim]
        let v = self.w_v.forward(input.clone()); // [batch_size, seq_len, output_embed_dim]

        let scores = q.matmul(k.transpose()) / self.dk_norm_factor;

        let context_length = input.shape().dims[1];
        let diag_mask: Tensor<B, 2, burn::tensor::Bool> =
            Tensor::<B, 2, burn::tensor::Bool>::tril_mask(
                Shape::from([context_length, context_length]),
                0,
                &input.device(),
            );
        // expand mask to [1, seq_len, seq_len]
        let diag_mask = diag_mask.unsqueeze::<3>();

        let masked_scores = scores.mask_fill(diag_mask, f32::NEG_INFINITY);
        let attn_weights = softmax(masked_scores, 2);

        let attn_weights = self.dropout.forward(attn_weights);

        // [batch_size, seq_len, output_embed_dim]
        attn_weights.matmul(v)
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::{Distribution, Shape};

    use super::*;
    use crate::TestBackend;

    #[test]
    fn test_custom_attention() {
        let config = CustomAttentionConfig::new(16, 32);
        let device = Default::default();
        let attention = config.init::<TestBackend>(&device, 0.0);
        let input =
            Tensor::<TestBackend, 3>::random([2, 10, 16], Distribution::Normal(0., 1.0), &device); // 2-batch, seq_len=10, input_dim=16
        let output = attention.forward(input);
        assert_eq!(output.shape(), Shape::from([2, 10, 32]));
    }
}
