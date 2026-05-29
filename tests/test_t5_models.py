import tempfile

import pytest
import torch

from life2lang.models import T5Config, T5Model, T5ForConditionalGeneration, T5Tokenizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_config():
    return T5Config(
        vocab_size=32128,
        d_model=64,
        d_kv=8,
        d_ff=128,
        num_layers=2,
        num_decoder_layers=2,
        num_heads=4,
        dropout_rate=0.0,
        decoder_start_token_id=0,
    )


@pytest.fixture(scope="module")
def t5_model(small_config):
    return T5Model(small_config)


@pytest.fixture(scope="module")
def t5_conditional(small_config):
    return T5ForConditionalGeneration(small_config)


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestT5Config:
    def test_default_instantiation(self):
        cfg = T5Config()
        assert cfg.vocab_size == 32128
        assert cfg.d_model == 512
        assert cfg.num_layers == 6

    def test_custom_instantiation(self, small_config):
        assert small_config.d_model == 64
        assert small_config.num_layers == 2
        assert small_config.num_decoder_layers == 2

    def test_attribute_aliases(self, small_config):
        assert small_config.hidden_size == small_config.d_model
        assert small_config.num_attention_heads == small_config.num_heads
        assert small_config.num_hidden_layers == small_config.num_layers

    def test_gated_act_parsing(self):
        cfg = T5Config(feed_forward_proj="gated-gelu")
        assert cfg.is_gated_act is True
        assert cfg.dense_act_fn == "gelu_new"

    def test_relu_act_parsing(self):
        cfg = T5Config(feed_forward_proj="relu")
        assert cfg.is_gated_act is False
        assert cfg.dense_act_fn == "relu"

    def test_invalid_feed_forward_proj(self):
        with pytest.raises(ValueError):
            T5Config(feed_forward_proj="invalid-act-fn-value")

    def test_save_and_load(self, small_config):
        with tempfile.TemporaryDirectory() as tmpdir:
            small_config.save_pretrained(tmpdir)
            loaded = T5Config.from_pretrained(tmpdir)
        assert loaded.d_model == small_config.d_model
        assert loaded.num_layers == small_config.num_layers
        assert loaded.vocab_size == small_config.vocab_size


# ---------------------------------------------------------------------------
# T5Model tests
# ---------------------------------------------------------------------------

class TestT5Model:
    def test_instantiation(self, t5_model):
        assert isinstance(t5_model, T5Model)

    def test_has_encoder_decoder(self, t5_model):
        assert hasattr(t5_model, "encoder")
        assert hasattr(t5_model, "decoder")

    def test_encoder_decoder_share_embeddings(self, t5_model):
        assert t5_model.encoder.embed_tokens.weight.data_ptr() == \
               t5_model.decoder.embed_tokens.weight.data_ptr()

    def test_forward_pass(self, t5_model, small_config):
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
        decoder_input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            outputs = t5_model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)

        assert outputs.last_hidden_state.shape == (batch_size, seq_len, small_config.d_model)

    def test_save_and_load_roundtrip(self, t5_model, small_config):
        with tempfile.TemporaryDirectory() as tmpdir:
            t5_model.save_pretrained(tmpdir)
            loaded = T5Model.from_pretrained(tmpdir)

        assert isinstance(loaded, T5Model)
        assert loaded.config.d_model == small_config.d_model
        assert loaded.config.num_layers == small_config.num_layers

    def test_loaded_weights_match(self, small_config):
        model = T5Model(small_config)
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            loaded = T5Model.from_pretrained(tmpdir)

        orig_sd = model.state_dict()
        loaded_sd = loaded.state_dict()
        assert set(orig_sd.keys()) == set(loaded_sd.keys())
        for key in orig_sd:
            assert torch.allclose(orig_sd[key], loaded_sd[key]), f"Mismatch at {key}"


# ---------------------------------------------------------------------------
# T5ForConditionalGeneration tests
# ---------------------------------------------------------------------------

class TestT5ForConditionalGeneration:
    def test_instantiation(self, t5_conditional):
        assert isinstance(t5_conditional, T5ForConditionalGeneration)

    def test_has_lm_head(self, t5_conditional, small_config):
        assert hasattr(t5_conditional, "lm_head")
        assert t5_conditional.lm_head.out_features == small_config.vocab_size

    def test_forward_without_labels(self, t5_conditional, small_config):
        batch_size, seq_len = 2, 8
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
        decoder_input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            outputs = t5_conditional(input_ids=input_ids, decoder_input_ids=decoder_input_ids)

        assert outputs.logits.shape == (batch_size, seq_len, small_config.vocab_size)
        assert outputs.loss is None

    def test_forward_with_labels_computes_loss(self, t5_conditional, small_config):
        batch_size, seq_len = 2, 8
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            outputs = t5_conditional(input_ids=input_ids, labels=labels)

        assert outputs.loss is not None
        assert outputs.loss.ndim == 0
        assert outputs.loss.item() > 0

    def test_labels_minus_100_are_ignored(self, t5_conditional, small_config):
        batch_size, seq_len = 2, 8
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))

        # Mix: first half ignored, second half normal
        mixed_labels = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
        all_normal_labels = mixed_labels.clone()
        mixed_labels[:, :seq_len // 2] = -100  # mask first half

        with torch.no_grad():
            loss_mixed = t5_conditional(input_ids=input_ids, labels=mixed_labels).loss
            loss_normal = t5_conditional(input_ids=input_ids, labels=all_normal_labels).loss

        assert loss_mixed.item() > 0.0
        assert loss_normal.item() > 0.0
        # Masking half the positions changes (typically increases) the loss per token
        assert loss_mixed.item() != loss_normal.item()

    def test_generate(self, t5_conditional, small_config):
        input_ids = torch.randint(0, small_config.vocab_size, (1, 5))

        with torch.no_grad():
            output_ids = t5_conditional.generate(input_ids, max_new_tokens=4)

        assert output_ids.ndim == 2
        assert output_ids.shape[0] == 1

    def test_save_and_load_roundtrip(self, t5_conditional, small_config):
        with tempfile.TemporaryDirectory() as tmpdir:
            t5_conditional.save_pretrained(tmpdir)
            loaded = T5ForConditionalGeneration.from_pretrained(tmpdir)

        assert isinstance(loaded, T5ForConditionalGeneration)
        assert loaded.config.vocab_size == small_config.vocab_size

    def test_config_preserved_after_save_load(self, t5_conditional, small_config):
        with tempfile.TemporaryDirectory() as tmpdir:
            t5_conditional.save_pretrained(tmpdir)
            loaded = T5ForConditionalGeneration.from_pretrained(tmpdir)

        for attr in ("d_model", "d_ff", "num_layers", "num_heads", "vocab_size"):
            assert getattr(loaded.config, attr) == getattr(small_config, attr)


# ---------------------------------------------------------------------------
# Tokenizer tests (requires network — marked slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestT5TokenizerFromPretrained:
    HF_MODEL_ID = "google-t5/t5-small"

    @pytest.fixture(scope="class")
    def tokenizer(self):
        return T5Tokenizer.from_pretrained(self.HF_MODEL_ID)

    def test_loads(self, tokenizer):
        assert isinstance(tokenizer, T5Tokenizer)

    def test_vocab_size(self, tokenizer):
        assert tokenizer.vocab_size == 32100

    def test_special_tokens(self, tokenizer):
        assert tokenizer.pad_token == "<pad>"
        assert tokenizer.eos_token == "</s>"
        assert tokenizer.unk_token == "<unk>"

    def test_encode_decode_roundtrip(self, tokenizer):
        text = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL"
        ids = tokenizer.encode(text, return_tensors="pt")
        decoded = tokenizer.decode(ids[0], skip_special_tokens=True)
        assert decoded.replace(" ", "").upper() in text.upper()

    def test_sentinel_tokens_present(self, tokenizer):
        for i in range(10):
            token = f"<extra_id_{i}>"
            assert tokenizer.convert_tokens_to_ids(token) != tokenizer.unk_token_id

    def test_save_and_reload(self, tokenizer):
        with tempfile.TemporaryDirectory() as tmpdir:
            tokenizer.save_pretrained(tmpdir)
            reloaded = T5Tokenizer.from_pretrained(tmpdir)
        assert reloaded.vocab_size == tokenizer.vocab_size
