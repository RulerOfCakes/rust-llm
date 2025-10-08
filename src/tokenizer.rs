use std::ops::Deref;

use tiktoken_rs::{CoreBPE, r50k_base};
pub struct GPT2Tokenizer(CoreBPE);

impl GPT2Tokenizer {
    pub fn new() -> Self {
        let bpe = r50k_base().unwrap();
        Self(bpe)
    }
}

impl Default for GPT2Tokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Deref for GPT2Tokenizer {
    type Target = CoreBPE;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpt2_tokenizer() {
        let tokenizer = GPT2Tokenizer::new();
        let input = "Akwirw ier";
        let tokens = tokenizer.encode_with_special_tokens(input);
        dbg!(&tokens);
        assert!(!tokens.is_empty());
        let output = tokenizer.decode(tokens).unwrap();
        assert_eq!(input, output);
    }
}
