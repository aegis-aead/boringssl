#include <openssl/aead.h>

#include <openssl/cipher.h>
#include <openssl/crypto.h>
#include <openssl/err.h>
#include <openssl/mem.h>
#include <openssl/span.h>

#include <assert.h>

#include "../fipsmodule/cipher/internal.h"
#include "../internal.h"

using namespace bssl;

#define AEGIS_128L_KEY_LEN 16
#define AEGIS_128L_NONCE_LEN 16
#define AEGIS_128L_TAG_LEN 16

#if defined(OPENSSL_X86_64) || defined(OPENSSL_AARCH64)

#ifdef OPENSSL_X86_64
#ifdef __clang__
#pragma clang attribute push(__attribute__((target("aes,avx"))), \
                             apply_to = function)
#elif defined(__GNUC__)
#pragma GCC target("aes,avx")
#endif

#include <wmmintrin.h>

typedef __m128i aes_block_t;
#define AES_BLOCK_XOR(A, B) _mm_xor_si128((A), (B))
#define AES_BLOCK_AND(A, B) _mm_and_si128((A), (B))
#define AES_BLOCK_LOAD(A) \
  _mm_loadu_si128((const aes_block_t *)(const void *)(A))
#define AES_BLOCK_LOAD_64x2(A, B) _mm_set_epi64x((A), (B))
#define AES_BLOCK_STORE(A, B) _mm_storeu_si128((aes_block_t *)(void *)(A), (B))
#define AES_ENC(A, B) _mm_aesenc_si128((A), (B))

#elif defined(OPENSSL_AARCH64)

#include <arm_neon.h>

#ifdef __clang__
#pragma clang attribute push(__attribute__((target("neon,crypto,aes"))), \
                             apply_to = function)
#elif defined(__GNUC__)
#pragma GCC target("+simd+crypto")
#endif
#ifndef __ARM_FEATURE_CRYPTO
#define __ARM_FEATURE_CRYPTO 1
#endif
#ifndef __ARM_FEATURE_AES
#define __ARM_FEATURE_AES 1
#endif

typedef uint8x16_t aes_block_t;
#define AES_BLOCK_XOR(A, B) veorq_u8((A), (B))
#define AES_BLOCK_AND(A, B) vandq_u8((A), (B))
#define AES_BLOCK_LOAD(A) vld1q_u8(A)
#define AES_BLOCK_LOAD_64x2(A, B) \
  vreinterpretq_u8_u64(vsetq_lane_u64((A), vmovq_n_u64(B), 1))
#define AES_BLOCK_STORE(A, B) vst1q_u8((A), (B))
#define AES_ENC(A, B) veorq_u8(vaesmcq_u8(vaeseq_u8(vmovq_n_u8(0), (A))), (B))

#else
#error "Unsupported architecture"
#endif

struct aead_aegis_128l_ctx {
  uint8_t key[AEGIS_128L_KEY_LEN];
};

typedef aes_block_t aegis_128l_state[8];

static_assert(sizeof(((EVP_AEAD_CTX *)NULL)->state) >=
                  sizeof(struct aead_aegis_128l_ctx),
              "AEAD state is too small");
static_assert(alignof(union evp_aead_ctx_st_state) >=
                  alignof(struct aead_aegis_128l_ctx),
              "AEAD state has insufficient alignment");

static int aead_aegis_128l_init(EVP_AEAD_CTX *ctx, const uint8_t *key,
                                size_t key_len, size_t tag_len) {
  if (key_len != AEGIS_128L_KEY_LEN) {
    OPENSSL_PUT_ERROR(CIPHER, CIPHER_R_BAD_KEY_LENGTH);
    return 0;
  }
  if (tag_len == EVP_AEAD_DEFAULT_TAG_LENGTH) {
    tag_len = AEGIS_128L_TAG_LEN;
  }
  if (tag_len != AEGIS_128L_TAG_LEN) {
    OPENSSL_PUT_ERROR(CIPHER, CIPHER_R_TAG_TOO_LARGE);
    return 0;
  }

  struct aead_aegis_128l_ctx *aegis_ctx =
      (struct aead_aegis_128l_ctx *)&ctx->state;
  OPENSSL_memcpy(aegis_ctx->key, key, key_len);
  ctx->tag_len = tag_len;

  return 1;
}

static void aead_aegis_128l_cleanup(EVP_AEAD_CTX *ctx) {}

static inline void aegis_128l_state_update(aes_block_t *const state,
                                           const aes_block_t d1,
                                           const aes_block_t d2) {
  aes_block_t tmp;

  tmp = state[7];
  state[7] = AES_ENC(state[6], state[7]);
  state[6] = AES_ENC(state[5], state[6]);
  state[5] = AES_ENC(state[4], state[5]);
  state[4] = AES_ENC(state[3], state[4]);
  state[3] = AES_ENC(state[2], state[3]);
  state[2] = AES_ENC(state[1], state[2]);
  state[1] = AES_ENC(state[0], state[1]);
  state[0] = AES_ENC(tmp, state[0]);

  state[0] = AES_BLOCK_XOR(state[0], d1);
  state[4] = AES_BLOCK_XOR(state[4], d2);
}

static void aegis_128l_state_init(const uint8_t *key, const uint8_t *nonce,
                                  size_t nonce_len, aes_block_t *const state) {
  static const uint8_t c0_[] = {0xdb, 0x3d, 0x18, 0x55, 0x6d, 0xc2, 0x2f, 0xf1,
                                0x20, 0x11, 0x31, 0x42, 0x73, 0xb5, 0x28, 0xdd};
  static const uint8_t c1_[] = {0x00, 0x01, 0x01, 0x02, 0x03, 0x05, 0x08, 0x0d,
                                0x15, 0x22, 0x37, 0x59, 0x90, 0xe9, 0x79, 0x62};
  const aes_block_t c0 = AES_BLOCK_LOAD(c0_);
  const aes_block_t c1 = AES_BLOCK_LOAD(c1_);

  uint8_t padded_nonce[AEGIS_128L_NONCE_LEN] = {0};
  assert(nonce_len <= sizeof padded_nonce);
  OPENSSL_memcpy(padded_nonce, nonce, nonce_len);

  aes_block_t k = AES_BLOCK_LOAD(key);
  aes_block_t n = AES_BLOCK_LOAD(padded_nonce);

  state[0] = AES_BLOCK_XOR(k, n);
  state[1] = c0;
  state[2] = c1;
  state[3] = c0;
  state[4] = AES_BLOCK_XOR(k, n);
  state[5] = AES_BLOCK_XOR(k, c1);
  state[6] = AES_BLOCK_XOR(k, c0);
  state[7] = AES_BLOCK_XOR(k, c1);

  for (int i = 0; i < 10; i++) {
    aegis_128l_state_update(state, n, k);
  }
}

static void aead_aegis_128l_tag(uint8_t *tag, size_t adlen, size_t mlen,
                                aes_block_t *const state) {
  aes_block_t tmp;

  tmp = AES_BLOCK_LOAD_64x2((uint64_t)mlen << 3, (uint64_t)adlen << 3);
  tmp = AES_BLOCK_XOR(tmp, state[2]);

  for (int i = 0; i < 7; i++) {
    aegis_128l_state_update(state, tmp, tmp);
  }

  tmp = AES_BLOCK_XOR(state[6], state[5]);
  tmp = AES_BLOCK_XOR(tmp, state[4]);
  tmp = AES_BLOCK_XOR(tmp, state[3]);
  tmp = AES_BLOCK_XOR(tmp, state[2]);
  tmp = AES_BLOCK_XOR(tmp, state[1]);
  tmp = AES_BLOCK_XOR(tmp, state[0]);

  AES_BLOCK_STORE(tag, tmp);
}

static void aead_aegis_128l_enc(uint8_t *const dst, const uint8_t *const src,
                                aes_block_t *const state) {
  aes_block_t msg0, msg1;
  aes_block_t tmp0, tmp1;

  msg0 = AES_BLOCK_LOAD(src);
  msg1 = AES_BLOCK_LOAD(src + 16);
  tmp0 = AES_BLOCK_XOR(msg0, state[6]);
  tmp0 = AES_BLOCK_XOR(tmp0, state[1]);
  tmp1 = AES_BLOCK_XOR(msg1, state[2]);
  tmp1 = AES_BLOCK_XOR(tmp1, state[5]);
  tmp0 = AES_BLOCK_XOR(tmp0, AES_BLOCK_AND(state[2], state[3]));
  tmp1 = AES_BLOCK_XOR(tmp1, AES_BLOCK_AND(state[6], state[7]));
  AES_BLOCK_STORE(dst, tmp0);
  AES_BLOCK_STORE(dst + 16, tmp1);

  aegis_128l_state_update(state, msg0, msg1);
}

static void aead_aegis_128l_dec(uint8_t *const dst, const uint8_t *const src,
                                aes_block_t *const state) {
  aes_block_t msg0, msg1;

  msg0 = AES_BLOCK_LOAD(src);
  msg1 = AES_BLOCK_LOAD(src + 16);
  msg0 = AES_BLOCK_XOR(msg0, state[6]);
  msg0 = AES_BLOCK_XOR(msg0, state[1]);
  msg1 = AES_BLOCK_XOR(msg1, state[2]);
  msg1 = AES_BLOCK_XOR(msg1, state[5]);
  msg0 = AES_BLOCK_XOR(msg0, AES_BLOCK_AND(state[2], state[3]));
  msg1 = AES_BLOCK_XOR(msg1, AES_BLOCK_AND(state[6], state[7]));
  AES_BLOCK_STORE(dst, msg0);
  AES_BLOCK_STORE(dst + 16, msg1);

  aegis_128l_state_update(state, msg0, msg1);
}

static void aegis_128l_absorb_ad(bssl::Span<const CRYPTO_IVEC> aadvecs,
                                 aegis_128l_state state) {
  alignas(16) uint8_t buf[32];
  bssl::iovec::ForEachBlockRange<32>(
      aadvecs,
      [&](const uint8_t *in, size_t len) {
        while (len >= 32) {
          aead_aegis_128l_enc(buf, in, state);
          in += 32;
          len -= 32;
        }
        return true;
      },
      [&](const uint8_t *in, size_t len) {
        while (len >= 32) {
          aead_aegis_128l_enc(buf, in, state);
          in += 32;
          len -= 32;
        }
        if (len > 0) {
          OPENSSL_memset(buf, 0, 32);
          OPENSSL_memcpy(buf, in, len);
          aead_aegis_128l_enc(buf, buf, state);
        }
        return true;
      });
}

static int aead_aegis_128l_sealv(const EVP_AEAD_CTX *ctx,
                                 bssl::Span<const CRYPTO_IOVEC> iovecs,
                                 bssl::Span<uint8_t> out_tag,
                                 size_t *out_tag_len,
                                 bssl::Span<const uint8_t> nonce,
                                 bssl::Span<const CRYPTO_IVEC> aadvecs) {
  const struct aead_aegis_128l_ctx *aegis_ctx =
      (struct aead_aegis_128l_ctx *)&ctx->state;

  if (out_tag.size() < ctx->tag_len) {
    OPENSSL_PUT_ERROR(CIPHER, CIPHER_R_BUFFER_TOO_SMALL);
    return 0;
  }
  if (nonce.size() > AEGIS_128L_NONCE_LEN) {
    OPENSSL_PUT_ERROR(CIPHER, CIPHER_R_UNSUPPORTED_NONCE_SIZE);
    return 0;
  }

  size_t ad_len = bssl::iovec::TotalLength(aadvecs);
  size_t in_len = bssl::iovec::TotalLength(iovecs);

  alignas(64) aegis_128l_state state;
  aegis_128l_state_init(aegis_ctx->key, nonce.data(), nonce.size(), state);

  aegis_128l_absorb_ad(aadvecs, state);

  bssl::iovec::ForEachBlockRange<32, /*WriteOut=*/true>(
      iovecs,
      [&](const uint8_t *in, uint8_t *out, size_t len) {
        while (len >= 32) {
          aead_aegis_128l_enc(out, in, state);
          in += 32;
          out += 32;
          len -= 32;
        }
        return true;
      },
      [&](const uint8_t *in, uint8_t *out, size_t len) {
        while (len >= 32) {
          aead_aegis_128l_enc(out, in, state);
          in += 32;
          out += 32;
          len -= 32;
        }
        if (len > 0) {
          alignas(16) uint8_t buf[32];
          OPENSSL_memset(buf, 0, 32);
          OPENSSL_memcpy(buf, in, len);
          aead_aegis_128l_enc(buf, buf, state);
          OPENSSL_memcpy(out, buf, len);
        }
        return true;
      });

  aead_aegis_128l_tag(out_tag.data(), ad_len, in_len, state);
  *out_tag_len = ctx->tag_len;

  OPENSSL_memset(state, 0, sizeof state);

  return 1;
}

static void aegis_128l_dec_partial(uint8_t *out, const uint8_t *in,
                                   size_t in_len, aegis_128l_state state) {
  alignas(16) uint8_t buf[32];
  OPENSSL_memset(buf, 0, 32);
  OPENSSL_memcpy(buf, in, in_len);
  aead_aegis_128l_dec(buf, buf, state);
  OPENSSL_memcpy(out, buf, in_len);
  OPENSSL_memset(buf, 0, in_len);
  state[0] = AES_BLOCK_XOR(state[0], AES_BLOCK_LOAD(buf));
  state[4] = AES_BLOCK_XOR(state[4], AES_BLOCK_LOAD(buf + 16));
}

static int aead_aegis_128l_openv_detached(
    const EVP_AEAD_CTX *ctx, bssl::Span<const CRYPTO_IOVEC> iovecs,
    bssl::Span<const uint8_t> nonce, bssl::Span<const uint8_t> in_tag,
    bssl::Span<const CRYPTO_IVEC> aadvecs) {
  const struct aead_aegis_128l_ctx *aegis_ctx =
      (struct aead_aegis_128l_ctx *)&ctx->state;

  if (nonce.size() > AEGIS_128L_NONCE_LEN) {
    OPENSSL_PUT_ERROR(CIPHER, CIPHER_R_UNSUPPORTED_NONCE_SIZE);
    return 0;
  }
  if (in_tag.size() != ctx->tag_len) {
    OPENSSL_PUT_ERROR(CIPHER, CIPHER_R_BAD_DECRYPT);
    return 0;
  }

  size_t ad_len = bssl::iovec::TotalLength(aadvecs);
  size_t in_len = bssl::iovec::TotalLength(iovecs);

  alignas(64) aegis_128l_state state;
  aegis_128l_state_init(aegis_ctx->key, nonce.data(), nonce.size(), state);

  aegis_128l_absorb_ad(aadvecs, state);

  bssl::iovec::ForEachBlockRange<32, /*WriteOut=*/true>(
      iovecs,
      [&](const uint8_t *in, uint8_t *out, size_t len) {
        while (len >= 32) {
          aead_aegis_128l_dec(out, in, state);
          in += 32;
          out += 32;
          len -= 32;
        }
        return true;
      },
      [&](const uint8_t *in, uint8_t *out, size_t len) {
        while (len >= 32) {
          aead_aegis_128l_dec(out, in, state);
          in += 32;
          out += 32;
          len -= 32;
        }
        if (len > 0) {
          aegis_128l_dec_partial(out, in, len, state);
        }
        return true;
      });

  uint8_t tag[AEGIS_128L_TAG_LEN];
  aead_aegis_128l_tag(tag, ad_len, in_len, state);

  OPENSSL_memset(state, 0, sizeof state);

  if (CRYPTO_memcmp(tag, in_tag.data(), ctx->tag_len) != 0) {
    OPENSSL_PUT_ERROR(CIPHER, CIPHER_R_BAD_DECRYPT);
    return 0;
  }

  return 1;
}

static const EVP_AEAD aead_aegis_128l = {
    AEGIS_128L_KEY_LEN,    // key length
    AEGIS_128L_NONCE_LEN,  // nonce length
    AEGIS_128L_TAG_LEN,    // overhead
    AEGIS_128L_TAG_LEN,    // max tag length

    aead_aegis_128l_init,
    nullptr,  // init_with_direction
    aead_aegis_128l_cleanup,
    nullptr,  // openv
    aead_aegis_128l_sealv,
    aead_aegis_128l_openv_detached,
    nullptr,  // get_iv
    nullptr,  // tag_len
};

#if defined(OPENSSL_X86_64) || defined(OPENSSL_AARCH64)
#ifdef __clang__
#pragma clang attribute pop
#endif
#endif

#endif

#ifdef OPENSSL_X86_64
const EVP_AEAD *EVP_aead_aegis_128l(void) {
  if (CRYPTO_is_AVX_capable() && CRYPTO_is_AESNI_capable()) {
    return &aead_aegis_128l;
  }
  return NULL;
}
#elif defined(OPENSSL_AARCH64)
const EVP_AEAD *EVP_aead_aegis_128l(void) {
  if (CRYPTO_is_NEON_capable()) {
    return &aead_aegis_128l;
  }
  return NULL;
}
#else
const EVP_AEAD *EVP_aead_aegis_128l(void) { return NULL; }
#endif
