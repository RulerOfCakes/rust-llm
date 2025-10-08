use std::path::Path;

use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    prelude::Backend,
    tensor::{Shape, Tensor, TensorData},
};
use tiktoken_rs::CoreBPE;

type Token = u32;
type TokenSequence = Vec<Token>;

#[derive(Clone, Debug)]
pub struct TextItem {
    pub input: TokenSequence,
    pub target: TokenSequence,
}

pub struct TextDataset {
    items: Vec<TextItem>,
}

impl Dataset<TextItem> for TextDataset {
    fn get(&self, index: usize) -> Option<TextItem> {
        self.items.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}

impl TextDataset {
    pub fn new(path: &Path, tokenizer: CoreBPE, max_length: usize, stride: usize) -> Self {
        let mut inputs = Vec::new();
        let mut targets = Vec::new();

        let text = std::fs::read_to_string(path).expect("Failed to read text file");
        let tokenized_inputs = tokenizer.encode_with_special_tokens(&text);
        for i in (0..tokenized_inputs.len() - 1).step_by(stride) {
            let end = (i + max_length).min(tokenized_inputs.len() - 1);
            inputs.push(tokenized_inputs[i..end].to_vec());
            targets.push(tokenized_inputs[i + 1..end + 1].to_vec());
        }

        Self {
            items: inputs
                .into_iter()
                .zip(targets)
                .map(|(input, target)| TextItem { input, target })
                .collect(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct TextBatch<B: Backend> {
    pub inputs: Tensor<B, 2>, // [batch_size, seq_len]
    pub targets: Tensor<B, 2>,
}

#[derive(Clone, Default)]
pub struct TextBatcher {}

// Input to the `Batcher` will be provided from the `Dataset` implementation when building the `DataLoader`.
impl<B: Backend> Batcher<B, TextItem, TextBatch<B>> for TextBatcher {
    fn batch(&self, items: Vec<TextItem>, device: &B::Device) -> TextBatch<B> {
        let inputs = items
            .iter()
            .map(|item| {
                TensorData::new(item.input.clone(), Shape::from([item.input.len()]))
                    .convert::<B::IntElem>()
            })
            .map(|data| Tensor::<B, 1>::from_data(data, device))
            .collect();

        let inputs = Tensor::stack(inputs, 0);

        let targets = items
            .iter()
            .map(|item| {
                TensorData::new(item.target.clone(), Shape::from([item.target.len()]))
                    .convert::<B::IntElem>()
            })
            .map(|data| Tensor::<B, 1>::from_data(data, device))
            .collect();
        let targets = Tensor::stack(targets, 0);
        TextBatch { inputs, targets }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::data::dataloader::DataLoaderBuilder;
    use tiktoken_rs::r50k_base;

    use crate::TestBackend;

    const SEED: u64 = 42;

    #[test]
    fn test_text_dataset_loader() {
        let tokenizer = r50k_base().unwrap();
        let dataset = TextDataset::new(Path::new("data/the-verdict.txt"), tokenizer, 128, 64);
        let batcher = TextBatcher::default();
        let dataloader = DataLoaderBuilder::<TestBackend, _, _>::new(batcher)
            .batch_size(4)
            .shuffle(SEED)
            .build(dataset);

        let sample = dataloader.iter().take(1).collect::<Vec<_>>();
        assert_eq!(sample.len(), 1);
        let input_tensor = &sample[0].inputs;
        let target_tensor = &sample[0].targets;
        assert_eq!(input_tensor.shape().dims(), [4, 128]);
        assert_eq!(target_tensor.shape().dims(), [4, 128]);
    }
}
