use burn::{
    module::Module,
    nn::{Dropout, DropoutConfig, Linear, LinearConfig},
    prelude::Backend,
    tensor::{Shape, Tensor, activation::softmax},
};

pub struct CustomAttentionConfig {
    input_dim: usize,
    output_dim: usize,
    num_heads: usize,
    dropout: f64,
}

impl CustomAttentionConfig {
    pub fn new(input_dim: usize, output_dim: usize, dropout: f64, num_heads: usize) -> Self {
        Self {
            input_dim,
            output_dim,
            num_heads,
            dropout,
        }
    }
    pub fn init<B: Backend>(&self, device: &B::Device) -> CustomAttention<B> {
        let w_q = LinearConfig::new(self.input_dim, self.output_dim)
            .with_bias(false)
            .init(device);
        let w_k = LinearConfig::new(self.input_dim, self.output_dim)
            .with_bias(false)
            .init(device);
        let w_v = LinearConfig::new(self.input_dim, self.output_dim)
            .with_bias(false)
            .init(device);
        let out_proj = LinearConfig::new(self.output_dim, self.output_dim)
            .with_bias(false)
            .init(device);
        let dropout = DropoutConfig::new(self.dropout).init();
        assert!(
            self.output_dim % self.num_heads == 0,
            "output_dim must be divisible by num_heads"
        );
        CustomAttention {
            dk_norm_factor: (self.output_dim as f32).sqrt(),
            num_heads: self.num_heads,
            head_dim: self.output_dim / self.num_heads,
            w_q,
            w_k,
            w_v,
            out_proj,
            dropout,
        }
    }
}

#[derive(Debug, Module)]
pub struct CustomAttention<B: Backend> {
    dk_norm_factor: f32,
    num_heads: usize,
    head_dim: usize,

    w_q: Linear<B>, // [input_embed_dim, output_embed_dim]
    w_k: Linear<B>,
    w_v: Linear<B>,

    // Although this is not strictly necessary, it is often used to project the output back to the desired dimension
    out_proj: Linear<B>, // [output_embed_dim, output_embed_dim]

    dropout: Dropout,
}

impl<B: Backend> CustomAttention<B> {
    // perform multi-head scaled dot-product attention (single head)
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let (batch_size, seq_len) = (input.shape().dims[0], input.shape().dims[1]);
        let q = self.w_q.forward(input.clone()); // [batch_size, seq_len, output_embed_dim]
        let k = self.w_k.forward(input.clone()); // [batch_size, seq_len, output_embed_dim]
        let v = self.w_v.forward(input.clone()); // [batch_size, seq_len, output_embed_dim]

        // change dim layout to [batch_size, seq_len, num_heads, head_dim]
        let mha_shape = [batch_size, seq_len, self.num_heads, self.head_dim];
        let q = q.reshape(mha_shape);
        let k = k.reshape(mha_shape);
        let v = v.reshape(mha_shape);

        // transpose dim 1 and 2 to get [batch_size, num_heads, seq_len, head_dim]
        let q = q.swap_dims(1, 2);
        let k = k.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);

        // To follow the original semantics of Q @ K^T, k needs to be transposed again on dim 2 and 3
        // This produces scores of shape [batch_size, num_heads, seq_len, seq_len]
        let scores = q.matmul(k.swap_dims(2, 3));

        let diag_mask: Tensor<B, 2, burn::tensor::Bool> =
            Tensor::<B, 2, burn::tensor::Bool>::tril_mask(
                Shape::from([seq_len, seq_len]),
                0,
                &input.device(),
            );
        // expand mask to [1, 1, seq_len, seq_len]
        let diag_mask = diag_mask.unsqueeze::<4>();

        let masked_scores = scores.mask_fill(diag_mask, f32::NEG_INFINITY);

        let attn_weights = softmax(masked_scores / self.dk_norm_factor, 2);
        let attn_weights = self.dropout.forward(attn_weights);

        let values = attn_weights.matmul(v); // [batch_size, num_heads, seq_len, head_dim]
        let values = values.swap_dims(1, 2); // Restore original layout to [batch_size, seq_len, num_heads, head_dim]

        // Finally, reshape to [batch_size, seq_len, output_embed_dim]
        self.out_proj.forward(values.reshape([
            batch_size,
            seq_len,
            self.num_heads * self.head_dim, // output_embed_dim
        ]))
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::{Distribution, Shape};

    use super::*;
    use crate::TestBackend;

    #[test]
    fn test_custom_attention() {
        let config = CustomAttentionConfig::new(16, 32, 0.1, 4);
        let device = Default::default();
        let attention = config.init::<TestBackend>(&device);
        let input =
            Tensor::<TestBackend, 3>::random([2, 10, 16], Distribution::Normal(0., 1.0), &device); // 2-batch, seq_len=10, input_dim=16
        let output = attention.forward(input);
        assert_eq!(output.shape(), Shape::from([2, 10, 32]));
    }
}
