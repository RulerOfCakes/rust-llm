pub mod data;
pub mod tokenizer;

#[cfg(test)]
type TestBackend = burn::backend::NdArray;
