import unittest
import jax
import numpy as np

from models.tokenizer import TokenizerVQVAE
from utils.nn import STTransformer, VectorQuantizer, PositionalEncoding, STBlock
from utils.preprocess import patchify, unpatchify
from train_tokenizer import create_training_functions


class TokenizerReproducibilityTest(unittest.TestCase):
    """Test reproducibility of tokenizer components and full model."""

    def setUp(self):
        super().setUp()
        self.seed = 42
        self.rng = jax.random.PRNGKey(self.seed)

        self.batch_size = 1
        self.seq_len = 4
        self.image_height = 32
        self.image_width = 32
        self.image_channels = 3
        self.patch_size = 4
        self.model_dim = 64
        self.latent_dim = 16
        self.num_latents = 64
        self.num_blocks = 1
        self.num_heads = 2
        self.dropout = 0.1
        self.codebook_dropout = 0.01

        self.rng, input_rng = jax.random.split(self.rng)
        self.test_videos = jax.random.uniform(
            input_rng,
            (
                self.batch_size,
                self.seq_len,
                self.image_height,
                self.image_width,
                self.image_channels,
            ),
        )

    def test_positional_encoding_reproducibility(self):
        """Test that positional encoding is deterministic."""
        dim = 64
        max_len = 100

        pe1 = PositionalEncoding(dim, max_len)
        pe2 = PositionalEncoding(dim, max_len)

        rng1, rng2 = jax.random.split(self.rng)
        x = jax.random.uniform(rng1, (2, 3, 50, dim))

        params1 = pe1.init(rng1, x)
        params2 = pe2.init(rng2, x)

        output1 = pe1.apply(params1, x)
        output2 = pe2.apply(params2, x)

        np.testing.assert_array_equal(output1, output2)

    def test_st_block_reproducibility(self):
        """Test that STBlock is reproducible."""
        dim = 64
        num_heads = 4
        dropout = 0.1

        block1 = STBlock(dim, num_heads, dropout)
        block2 = STBlock(dim, num_heads, dropout)

        x = jax.random.uniform(self.rng, (2, 3, 16, dim))

        params1 = block1.init(self.rng, x)
        params2 = block2.init(self.rng, x)

        rng_dropout = jax.random.PRNGKey(123)
        output1 = block1.apply(params1, x, rngs={"dropout": rng_dropout})
        output2 = block2.apply(params2, x, rngs={"dropout": rng_dropout})

        np.testing.assert_array_equal(output1, output2)

    def test_st_block_forward_pass_reproducibility(self):
        """Test that STBlock forward pass is reproducible across multiple calls."""
        dim = 64
        num_heads = 4
        dropout = 0.1

        block = STBlock(dim, num_heads, dropout)

        rng = jax.random.PRNGKey(self.seed)
        x = jax.random.uniform(rng, (2, 3, 16, dim))
        params = block.init(rng, x)

        rng_dropout = jax.random.PRNGKey(123)
        outputs = []
        for _ in range(3):
            output = block.apply(params, x, rngs={"dropout": rng_dropout})
            outputs.append(output)

        for i in range(1, len(outputs)):
            np.testing.assert_array_equal(outputs[0], outputs[i])

    def test_st_transformer_reproducibility(self):
        """Test that STTransformer is reproducible."""
        model_dim = 64
        out_dim = 32
        num_blocks = 2
        num_heads = 4
        dropout = 0.1

        transformer1 = STTransformer(model_dim, out_dim, num_blocks, num_heads, dropout)
        transformer2 = STTransformer(model_dim, out_dim, num_blocks, num_heads, dropout)

        x = jax.random.uniform(self.rng, (2, 3, 16, 48))

        params1 = transformer1.init(self.rng, x)
        params2 = transformer2.init(self.rng, x)

        rng_dropout = jax.random.PRNGKey(123)
        output1 = transformer1.apply(params1, x, rngs={"dropout": rng_dropout})
        output2 = transformer2.apply(params2, x, rngs={"dropout": rng_dropout})

        np.testing.assert_array_equal(output1, output2)

    def test_vector_quantizer_reproducibility(self):
        """Test that VectorQuantizer is reproducible."""
        latent_dim = 32
        num_latents = 256
        dropout = 0.01

        vq1 = VectorQuantizer(latent_dim, num_latents, dropout)
        vq2 = VectorQuantizer(latent_dim, num_latents, dropout)

        x = jax.random.uniform(self.rng, (100, latent_dim))

        params1 = vq1.init(self.rng, x, training=True)
        params2 = vq2.init(self.rng, x, training=True)

        rng_dropout = jax.random.PRNGKey(123)
        output1 = vq1.apply(params1, x, training=True, rngs={"dropout": rng_dropout})
        output2 = vq2.apply(params2, x, training=True, rngs={"dropout": rng_dropout})

        for o1, o2 in zip(output1, output2):
            np.testing.assert_array_equal(o1, o2)

    def test_vector_quantizer_codebook_initialization_reproducibility(self):
        """Test that VectorQuantizer codebook initialization is reproducible."""
        latent_dim = 32
        num_latents = 256
        dropout = 0.01

        rng = jax.random.PRNGKey(self.seed)
        codebooks = []
        for i in range(3):
            vq = VectorQuantizer(latent_dim, num_latents, dropout)
            x = jax.random.uniform(rng, (100, latent_dim))
            params = vq.init(rng, x, training=True)
            codebook = params["params"]["codebook"]
            codebooks.append(codebook)

        for i in range(1, len(codebooks)):
            np.testing.assert_array_equal(codebooks[0], codebooks[i])

    def test_patchify_unpatchify_reproducibility(self):
        """Test that patchify/unpatchify operations are deterministic."""
        videos = self.test_videos
        patch_size = self.patch_size

        patches1 = patchify(videos, patch_size)
        patches2 = patchify(videos, patch_size)

        np.testing.assert_array_equal(patches1, patches2)

        H, W = videos.shape[2:4]
        recon1 = unpatchify(patches1, patch_size, H, W)
        recon2 = unpatchify(patches2, patch_size, H, W)

        np.testing.assert_array_equal(recon1, recon2)

    def test_tokenizer_vq_encode_reproducibility(self):
        """Test that tokenizer vq_encode is reproducible."""
        tokenizer1 = TokenizerVQVAE(
            in_dim=self.image_channels,
            model_dim=self.model_dim,
            latent_dim=self.latent_dim,
            num_latents=self.num_latents,
            patch_size=self.patch_size,
            num_blocks=self.num_blocks,
            num_heads=self.num_heads,
            dropout=self.dropout,
            codebook_dropout=self.codebook_dropout,
        )
        tokenizer2 = TokenizerVQVAE(
            in_dim=self.image_channels,
            model_dim=self.model_dim,
            latent_dim=self.latent_dim,
            num_latents=self.num_latents,
            patch_size=self.patch_size,
            num_blocks=self.num_blocks,
            num_heads=self.num_heads,
            dropout=self.dropout,
            codebook_dropout=self.codebook_dropout,
        )

        params1 = tokenizer1.init(self.rng, {"videos": self.test_videos})
        params2 = tokenizer2.init(self.rng, {"videos": self.test_videos})

        rng_dropout = jax.random.PRNGKey(123)
        output1 = tokenizer1.apply(
            params1,
            self.test_videos,
            method=tokenizer1.vq_encode,
            training=True,
            rngs={"dropout": rng_dropout},
        )
        output2 = tokenizer2.apply(
            params2,
            self.test_videos,
            method=tokenizer2.vq_encode,
            training=True,
            rngs={"dropout": rng_dropout},
        )

        for key in output1.keys():
            np.testing.assert_array_equal(output1[key], output2[key])

    def test_full_tokenizer_reproducibility(self):
        """Test that full tokenizer forward pass is reproducible."""
        tokenizer1 = TokenizerVQVAE(
            in_dim=self.image_channels,
            model_dim=self.model_dim,
            latent_dim=self.latent_dim,
            num_latents=self.num_latents,
            patch_size=self.patch_size,
            num_blocks=self.num_blocks,
            num_heads=self.num_heads,
            dropout=self.dropout,
            codebook_dropout=self.codebook_dropout,
        )
        tokenizer2 = TokenizerVQVAE(
            in_dim=self.image_channels,
            model_dim=self.model_dim,
            latent_dim=self.latent_dim,
            num_latents=self.num_latents,
            patch_size=self.patch_size,
            num_blocks=self.num_blocks,
            num_heads=self.num_heads,
            dropout=self.dropout,
            codebook_dropout=self.codebook_dropout,
        )

        params1 = tokenizer1.init(self.rng, {"videos": self.test_videos})
        params2 = tokenizer2.init(self.rng, {"videos": self.test_videos})

        rng_dropout = jax.random.PRNGKey(123)
        output1 = tokenizer1.apply(
            params1,
            {"videos": self.test_videos},
            training=True,
            rngs={"dropout": rng_dropout},
        )
        output2 = tokenizer2.apply(
            params2,
            {"videos": self.test_videos},
            training=True,
            rngs={"dropout": rng_dropout},
        )

        for key in output1.keys():
            np.testing.assert_array_equal(output1[key], output2[key])

    def test_tokenizer_loss_reproducibility(self):
        """Test that tokenizer loss computation is reproducible."""

        class MockArgs:
            def __init__(self):
                self.vq_beta = 0.25
                self.num_latents = 64
                self.log_gradients = False

        args = MockArgs()

        tokenizer_loss_fn, _ = create_training_functions(args)

        tokenizer = TokenizerVQVAE(
            in_dim=self.image_channels,
            model_dim=self.model_dim,
            latent_dim=self.latent_dim,
            num_latents=self.num_latents,
            patch_size=self.patch_size,
            num_blocks=self.num_blocks,
            num_heads=self.num_heads,
            dropout=self.dropout,
            codebook_dropout=self.codebook_dropout,
        )

        params1 = tokenizer.init(self.rng, {"videos": self.test_videos})
        params2 = tokenizer.init(self.rng, {"videos": self.test_videos})

        rng_input = jax.random.PRNGKey(456)
        inputs = {"videos": self.test_videos, "rng": rng_input}

        class MockState:
            def __init__(self, apply_fn):
                self.apply_fn = apply_fn

        state1 = MockState(tokenizer.apply)
        state2 = MockState(tokenizer.apply)

        loss1, (recon1, metrics1) = tokenizer_loss_fn(params1, state1, inputs)
        loss2, (recon2, metrics2) = tokenizer_loss_fn(params2, state2, inputs)

        np.testing.assert_array_equal(loss1, loss2)
        np.testing.assert_array_equal(recon1, recon2)
        for key in metrics1.keys():
            np.testing.assert_array_equal(metrics1[key], metrics2[key])

    def test_tokenizer_gradient_reproducibility(self):
        """Test that gradients are reproducible."""

        class MockArgs:
            def __init__(self):
                self.vq_beta = 0.25
                self.num_latents = 64
                self.log_gradients = False

        args = MockArgs()

        tokenizer_loss_fn, _ = create_training_functions(args)

        tokenizer = TokenizerVQVAE(
            in_dim=self.image_channels,
            model_dim=self.model_dim,
            latent_dim=self.latent_dim,
            num_latents=self.num_latents,
            patch_size=self.patch_size,
            num_blocks=self.num_blocks,
            num_heads=self.num_heads,
            dropout=self.dropout,
            codebook_dropout=self.codebook_dropout,
        )

        rng = jax.random.PRNGKey(self.seed)
        test_videos = jax.random.uniform(
            rng,
            (
                self.batch_size,
                self.seq_len,
                self.image_height,
                self.image_width,
                self.image_channels,
            ),
        )
        params = tokenizer.init(rng, {"videos": test_videos})

        rng_input = jax.random.PRNGKey(456)
        inputs = {"videos": test_videos, "rng": rng_input}

        class MockState:
            def __init__(self, apply_fn):
                self.apply_fn = apply_fn

        state = MockState(tokenizer.apply)

        grad_fn = jax.value_and_grad(tokenizer_loss_fn, has_aux=True, allow_int=True)
        (loss1, (recon1, metrics1)), grads1 = grad_fn(params, state, inputs)
        (loss2, (recon2, metrics2)), grads2 = grad_fn(params, state, inputs)

        np.testing.assert_array_equal(loss1, loss2)
        np.testing.assert_array_equal(recon1, recon2)
        for key in metrics1.keys():
            np.testing.assert_array_equal(metrics1[key], metrics2[key])

        jax.tree.map(lambda x, y: np.testing.assert_array_equal(x, y), grads1, grads2)

    def test_training_step_reproducibility(self):
        """Test that training step is reproducible."""

        class MockArgs:
            def __init__(self):
                self.vq_beta = 0.25
                self.num_latents = 64
                self.log_gradients = False

        args = MockArgs()

        _, train_step = create_training_functions(args)

        tokenizer = TokenizerVQVAE(
            in_dim=self.image_channels,
            model_dim=self.model_dim,
            latent_dim=self.latent_dim,
            num_latents=self.num_latents,
            patch_size=self.patch_size,
            num_blocks=self.num_blocks,
            num_heads=self.num_heads,
            dropout=self.dropout,
            codebook_dropout=self.codebook_dropout,
        )

        params1 = tokenizer.init(self.rng, {"videos": self.test_videos})
        params2 = tokenizer.init(self.rng, {"videos": self.test_videos})

        import optax

        tx = optax.adamw(learning_rate=1e-4)
        from flax.training.train_state import TrainState

        state1 = TrainState.create(apply_fn=tokenizer.apply, params=params1, tx=tx)
        state2 = TrainState.create(apply_fn=tokenizer.apply, params=params2, tx=tx)

        rng_input = jax.random.PRNGKey(456)
        inputs = {"videos": self.test_videos, "rng": rng_input}

        _, loss1, recon1, metrics1 = train_step(state1, inputs)
        _, loss2, recon2, metrics2 = train_step(state2, inputs)

        np.testing.assert_array_equal(loss1, loss2)
        np.testing.assert_array_equal(recon1, recon2)
        for key in metrics1.keys():
            np.testing.assert_array_equal(metrics1[key], metrics2[key])

    def test_multiple_forward_passes_reproducibility(self):
        """Test that multiple forward passes with same seed are reproducible."""
        tokenizer = TokenizerVQVAE(
            in_dim=self.image_channels,
            model_dim=self.model_dim,
            latent_dim=self.latent_dim,
            num_latents=self.num_latents,
            patch_size=self.patch_size,
            num_blocks=self.num_blocks,
            num_heads=self.num_heads,
            dropout=self.dropout,
            codebook_dropout=self.codebook_dropout,
        )

        params = tokenizer.init(self.rng, {"videos": self.test_videos})

        base_rng = jax.random.PRNGKey(123)
        outputs = []
        for i in range(3):
            rng_dropout = jax.random.fold_in(base_rng, i)
            output = tokenizer.apply(
                params,
                {"videos": self.test_videos},
                training=True,
                rngs={"dropout": rng_dropout},
            )
            outputs.append(output)

        outputs2 = []
        for i in range(3):
            rng_dropout = jax.random.fold_in(base_rng, i)
            output = tokenizer.apply(
                params,
                {"videos": self.test_videos},
                training=True,
                rngs={"dropout": rng_dropout},
            )
            outputs2.append(output)

        for i in range(3):
            for key in outputs[i].keys():
                np.testing.assert_array_equal(outputs[i][key], outputs2[i][key])

    def test_tokenizer_indices_consistency(self):
        """Test that tokenizer produces consistent indices for same input."""
        tokenizer = TokenizerVQVAE(
            in_dim=self.image_channels,
            model_dim=self.model_dim,
            latent_dim=self.latent_dim,
            num_latents=self.num_latents,
            patch_size=self.patch_size,
            num_blocks=self.num_blocks,
            num_heads=self.num_heads,
            dropout=0.0,
            codebook_dropout=0.0,
        )

        rng = jax.random.PRNGKey(self.seed)
        test_videos = jax.random.uniform(
            rng,
            (
                self.batch_size,
                self.seq_len,
                self.image_height,
                self.image_width,
                self.image_channels,
            ),
        )
        params = tokenizer.init(rng, {"videos": test_videos})

        indices_list = []
        for _ in range(3):
            output = tokenizer.apply(
                params, test_videos, method=tokenizer.vq_encode, training=True
            )
            indices_list.append(output["indices"])

        for i in range(1, len(indices_list)):
            np.testing.assert_array_equal(indices_list[0], indices_list[i])

    def test_different_dropout_rngs_produce_different_outputs(self):
        """Test that different dropout RNGs produce different outputs (sanity check)."""
        tokenizer = TokenizerVQVAE(
            in_dim=self.image_channels,
            model_dim=self.model_dim,
            latent_dim=self.latent_dim,
            num_latents=self.num_latents,
            patch_size=self.patch_size,
            num_blocks=self.num_blocks,
            num_heads=self.num_heads,
            dropout=self.dropout,
            codebook_dropout=self.codebook_dropout,
        )

        params = tokenizer.init(self.rng, {"videos": self.test_videos})

        rng1 = jax.random.PRNGKey(123)
        rng2 = jax.random.PRNGKey(456)

        output1 = tokenizer.apply(
            params, {"videos": self.test_videos}, training=True, rngs={"dropout": rng1}
        )
        output2 = tokenizer.apply(
            params, {"videos": self.test_videos}, training=True, rngs={"dropout": rng2}
        )

        for key in output1.keys():
            if key in ["recon", "z_q", "z", "emb"]:
                with self.assertRaises(AssertionError):
                    np.testing.assert_array_equal(output1[key], output2[key])


if __name__ == "__main__":
    unittest.main()
