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

#define AEGIS_128X2_KEY_LEN 16
#define AEGIS_128X2_NONCE_LEN 16
#define AEGIS_128X2_TAG_LEN 16

#define AES_BLOCK_LENGTH 32
#define RATE (2 * AES_BLOCK_LENGTH)

#if defined(OPENSSL_X86_64) || defined(OPENSSL_AARCH64)

#ifdef OPENSSL_X86_64
#ifdef __clang__
#pragma clang attribute push(__attribute__((target("vaes,avx2"))), \
                             apply_to = function)
#elif defined(__GNUC__)
#pragma GCC target("vaes,avx2")
#endif

#include <immintrin.h>

typedef __m256i aes_block_t;

#define AES_BLOCK_XOR(A, B) _mm256_xor_si256((A), (B))
#define AES_BLOCK_AND(A, B) _mm256_and_si256((A), (B))
#define AES_BLOCK_LOAD128_BROADCAST(A) \
  _mm256_broadcastsi128_si256(_mm_loadu_si128((const void *)(A)))
#define AES_BLOCK_LOAD(A) \
  _mm256_loadu_si256((const aes_block_t *)(const void *)(A))
#define AES_BLOCK_LOAD_64x2(A, B) \
  _mm256_broadcastsi128_si256(_mm_set_epi64x((A), (B)))
#define AES_BLOCK_STORE(A, B) \
  _mm256_storeu_si256((aes_block_t *)(void *)(A), (B))
#define AES_ENC(A, B) _mm256_aesenc_epi128((A), (B))

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

typedef struct {
  uint8x16_t b0;
  uint8x16_t b1;
} aes_block_t;

static inline aes_block_t AES_BLOCK_XOR(const aes_block_t a,
                                        const aes_block_t b) {
  return (aes_block_t){veorq_u8(a.b0, b.b0), veorq_u8(a.b1, b.b1)};
}

static inline aes_block_t AES_BLOCK_AND(const aes_block_t a,
                                        const aes_block_t b) {
  return (aes_block_t){vandq_u8(a.b0, b.b0), vandq_u8(a.b1, b.b1)};
}

static inline aes_block_t AES_BLOCK_LOAD(const uint8_t *a) {
  return (aes_block_t){vld1q_u8(a), vld1q_u8(a + 16)};
}

static inline aes_block_t AES_BLOCK_LOAD_64x2(uint64_t a, uint64_t b) {
  const uint8x16_t t =
      vreinterpretq_u8_u64(vsetq_lane_u64((a), vmovq_n_u64(b), 1));
  return (aes_block_t){t, t};
}
static inline void AES_BLOCK_STORE(uint8_t *a, const aes_block_t b) {
  vst1q_u8(a, b.b0);
  vst1q_u8(a + 16, b.b1);
}

static inline aes_block_t AES_ENC(const aes_block_t a, const aes_block_t b) {
  return (aes_block_t){
      veorq_u8(vaesmcq_u8(vaeseq_u8(vmovq_n_u8(0), (a.b0))), (b.b0)),
      veorq_u8(vaesmcq_u8(vaeseq_u8(vmovq_n_u8(0), (a.b1))), (b.b1))};
}

#else
#error "Unsupported architecture"
#endif

struct aead_aegis_128x2_ctx {
  uint8_t key[AEGIS_128X2_KEY_LEN];
};

typedef aes_block_t aegis_128x2_state[8];

static_assert(sizeof(((EVP_AEAD_CTX *)NULL)->state) >=
                  sizeof(struct aead_aegis_128x2_ctx),
              "AEAD state is too small");
static_assert(alignof(union evp_aead_ctx_st_state) >=
                  alignof(struct aead_aegis_128x2_ctx),
              "AEAD state has insufficient alignment");

static int aead_aegis_128x2_init(EVP_AEAD_CTX *ctx, const uint8_t *key,
                                 size_t key_len, size_t tag_len) {
  if (key_len != AEGIS_128X2_KEY_LEN) {
    OPENSSL_PUT_ERROR(CIPHER, CIPHER_R_BAD_KEY_LENGTH);
    return 0;
  }
  if (tag_len == EVP_AEAD_DEFAULT_TAG_LENGTH) {
    tag_len = AEGIS_128X2_TAG_LEN;
  }
  if (tag_len != AEGIS_128X2_TAG_LEN) {
    OPENSSL_PUT_ERROR(CIPHER, CIPHER_R_TAG_TOO_LARGE);
    return 0;
  }

  struct aead_aegis_128x2_ctx *aegis_ctx =
      (struct aead_aegis_128x2_ctx *)&ctx->state;
  OPENSSL_memcpy(aegis_ctx->key, key, key_len);
  ctx->tag_len = tag_len;

  return 1;
}

static void aead_aegis_128x2_cleanup(EVP_AEAD_CTX *ctx) {}

static inline void aegis_128x2_state_update(aes_block_t *const state,
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

static void aegis_128x2_state_init(const uint8_t *key, const uint8_t *nonce,
                                   size_t nonce_len, aes_block_t *const state) {
  static const uint8_t c0_[] = {0xdb, 0x3d, 0x18, 0x55, 0x6d, 0xc2, 0x2f, 0xf1,
                                0x20, 0x11, 0x31, 0x42, 0x73, 0xb5, 0x28, 0xdd,
                                0xdb, 0x3d, 0x18, 0x55, 0x6d, 0xc2, 0x2f, 0xf1,
                                0x20, 0x11, 0x31, 0x42, 0x73, 0xb5, 0x28, 0xdd};
  static const uint8_t c1_[] = {
      0x00, 0x01, 0x01, 0x02, 0x03, 0x05, 0x08, 0x0d, 0x15, 0x22, 0x37,
      0x59, 0x90, 0xe9, 0x79, 0x62, 0x00, 0x01, 0x01, 0x02, 0x03, 0x05,
      0x08, 0x0d, 0x15, 0x22, 0x37, 0x59, 0x90, 0xe9, 0x79, 0x62,
  };
  const aes_block_t c0 = AES_BLOCK_LOAD(c0_);
  const aes_block_t c1 = AES_BLOCK_LOAD(c1_);
  uint8_t context_bytes[AES_BLOCK_LENGTH];
  aes_block_t context;

  uint8_t padded_nonce[AEGIS_128X2_NONCE_LEN] = {0};
  assert(nonce_len <= sizeof padded_nonce);
  OPENSSL_memcpy(padded_nonce, nonce, nonce_len);

  uint8_t key_x[AES_BLOCK_LENGTH];
  uint8_t nonce_x[AES_BLOCK_LENGTH];
  memcpy(key_x, key, AEGIS_128X2_KEY_LEN);
  memcpy(key_x + AEGIS_128X2_KEY_LEN, key, AEGIS_128X2_KEY_LEN);
  memcpy(nonce_x, padded_nonce, AEGIS_128X2_NONCE_LEN);
  memcpy(nonce_x + AEGIS_128X2_NONCE_LEN, padded_nonce, AEGIS_128X2_NONCE_LEN);

  aes_block_t k = AES_BLOCK_LOAD(key_x);
  aes_block_t n = AES_BLOCK_LOAD(nonce_x);

  memset(context_bytes, 0, sizeof context_bytes);
  context_bytes[0 * 16] = 0x00;
  context_bytes[0 * 16 + 1] = 0x01;
  context_bytes[1 * 16] = 0x01;
  context_bytes[1 * 16 + 1] = 0x01;
  context = AES_BLOCK_LOAD(context_bytes);

  state[0] = AES_BLOCK_XOR(k, n);
  state[1] = c0;
  state[2] = c1;
  state[3] = c0;
  state[4] = AES_BLOCK_XOR(k, n);
  state[5] = AES_BLOCK_XOR(k, c1);
  state[6] = AES_BLOCK_XOR(k, c0);
  state[7] = AES_BLOCK_XOR(k, c1);

  for (int i = 0; i < 10; i++) {
    state[3] = AES_BLOCK_XOR(state[3], context);
    state[7] = AES_BLOCK_XOR(state[7], context);
    aegis_128x2_state_update(state, n, k);
  }
}

static void aead_aegis_128x2_tag(uint8_t *tag, size_t adlen, size_t mlen,
                                 aes_block_t *const state) {
  aes_block_t tmp;

  tmp = AES_BLOCK_LOAD_64x2((uint64_t)mlen << 3, (uint64_t)adlen << 3);
  tmp = AES_BLOCK_XOR(tmp, state[2]);

  for (int i = 0; i < 7; i++) {
    aegis_128x2_state_update(state, tmp, tmp);
  }

  uint8_t mac_multi[AES_BLOCK_LENGTH];
  tmp = AES_BLOCK_XOR(state[6], AES_BLOCK_XOR(state[5], state[4]));
  tmp = AES_BLOCK_XOR(tmp, AES_BLOCK_XOR(state[3], state[2]));
  tmp = AES_BLOCK_XOR(tmp, AES_BLOCK_XOR(state[1], state[0]));
  AES_BLOCK_STORE(mac_multi, tmp);
  for (size_t i = 0; i < 16; i++) {
    tag[i] = mac_multi[i] ^ mac_multi[1 * 16 + i];
  }
}

static void aead_aegis_128x2_enc(uint8_t *const dst, const uint8_t *const src,
                                 aes_block_t *const state) {
  aes_block_t msg0, msg1;
  aes_block_t tmp0, tmp1;

  msg0 = AES_BLOCK_LOAD(src);
  msg1 = AES_BLOCK_LOAD(src + AES_BLOCK_LENGTH);
  tmp0 = AES_BLOCK_XOR(msg0, state[6]);
  tmp0 = AES_BLOCK_XOR(tmp0, state[1]);
  tmp1 = AES_BLOCK_XOR(msg1, state[2]);
  tmp1 = AES_BLOCK_XOR(tmp1, state[5]);
  tmp0 = AES_BLOCK_XOR(tmp0, AES_BLOCK_AND(state[2], state[3]));
  tmp1 = AES_BLOCK_XOR(tmp1, AES_BLOCK_AND(state[6], state[7]));
  AES_BLOCK_STORE(dst, tmp0);
  AES_BLOCK_STORE(dst + AES_BLOCK_LENGTH, tmp1);

  aegis_128x2_state_update(state, msg0, msg1);
}

static void aead_aegis_128x2_dec(uint8_t *const dst, const uint8_t *const src,
                                 aes_block_t *const state) {
  aes_block_t msg0, msg1;

  msg0 = AES_BLOCK_LOAD(src);
  msg1 = AES_BLOCK_LOAD(src + AES_BLOCK_LENGTH);
  msg0 = AES_BLOCK_XOR(msg0, state[6]);
  msg0 = AES_BLOCK_XOR(msg0, state[1]);
  msg1 = AES_BLOCK_XOR(msg1, state[2]);
  msg1 = AES_BLOCK_XOR(msg1, state[5]);
  msg0 = AES_BLOCK_XOR(msg0, AES_BLOCK_AND(state[2], state[3]));
  msg1 = AES_BLOCK_XOR(msg1, AES_BLOCK_AND(state[6], state[7]));
  AES_BLOCK_STORE(dst, msg0);
  AES_BLOCK_STORE(dst + AES_BLOCK_LENGTH, msg1);

  aegis_128x2_state_update(state, msg0, msg1);
}

static void aegis_128x2_absorb_ad(bssl::Span<const CRYPTO_IVEC> aadvecs,
                                  aegis_128x2_state state) {
  alignas(AES_BLOCK_LENGTH) uint8_t buf[RATE];
  bssl::iovec::ForEachBlockRange<RATE>(
      aadvecs,
      [&](const uint8_t *in, size_t len) {
        while (len >= RATE) {
          aead_aegis_128x2_enc(buf, in, state);
          in += RATE;
          len -= RATE;
        }
        return true;
      },
      [&](const uint8_t *in, size_t len) {
        while (len >= RATE) {
          aead_aegis_128x2_enc(buf, in, state);
          in += RATE;
          len -= RATE;
        }
        if (len > 0) {
          OPENSSL_memset(buf, 0, RATE);
          OPENSSL_memcpy(buf, in, len);
          aead_aegis_128x2_enc(buf, buf, state);
        }
        return true;
      });
}

static int aead_aegis_128x2_sealv(const EVP_AEAD_CTX *ctx,
                                  bssl::Span<const CRYPTO_IOVEC> iovecs,
                                  bssl::Span<uint8_t> out_tag,
                                  size_t *out_tag_len,
                                  bssl::Span<const uint8_t> nonce,
                                  bssl::Span<const CRYPTO_IVEC> aadvecs) {
  const struct aead_aegis_128x2_ctx *aegis_ctx =
      (struct aead_aegis_128x2_ctx *)&ctx->state;

  if (out_tag.size() < ctx->tag_len) {
    OPENSSL_PUT_ERROR(CIPHER, CIPHER_R_BUFFER_TOO_SMALL);
    return 0;
  }
  if (nonce.size() > AEGIS_128X2_NONCE_LEN) {
    OPENSSL_PUT_ERROR(CIPHER, CIPHER_R_UNSUPPORTED_NONCE_SIZE);
    return 0;
  }

  size_t ad_len = bssl::iovec::TotalLength(aadvecs);
  size_t in_len = bssl::iovec::TotalLength(iovecs);

  alignas(64) aegis_128x2_state state;
  aegis_128x2_state_init(aegis_ctx->key, nonce.data(), nonce.size(), state);

  aegis_128x2_absorb_ad(aadvecs, state);

  bssl::iovec::ForEachBlockRange<RATE, /*WriteOut=*/true>(
      iovecs,
      [&](const uint8_t *in, uint8_t *out, size_t len) {
        while (len >= RATE) {
          aead_aegis_128x2_enc(out, in, state);
          in += RATE;
          out += RATE;
          len -= RATE;
        }
        return true;
      },
      [&](const uint8_t *in, uint8_t *out, size_t len) {
        while (len >= RATE) {
          aead_aegis_128x2_enc(out, in, state);
          in += RATE;
          out += RATE;
          len -= RATE;
        }
        if (len > 0) {
          alignas(AES_BLOCK_LENGTH) uint8_t buf[RATE];
          OPENSSL_memset(buf, 0, RATE);
          OPENSSL_memcpy(buf, in, len);
          aead_aegis_128x2_enc(buf, buf, state);
          OPENSSL_memcpy(out, buf, len);
        }
        return true;
      });

  aead_aegis_128x2_tag(out_tag.data(), ad_len, in_len, state);
  *out_tag_len = ctx->tag_len;

  OPENSSL_memset(state, 0, sizeof state);

  return 1;
}

static void aegis_128x2_dec_partial(uint8_t *out, const uint8_t *in,
                                    size_t in_len, aegis_128x2_state state) {
  alignas(AES_BLOCK_LENGTH) uint8_t buf[RATE];
  OPENSSL_memset(buf, 0, RATE);
  OPENSSL_memcpy(buf, in, in_len);
  aead_aegis_128x2_dec(buf, buf, state);
  OPENSSL_memcpy(out, buf, in_len);
  OPENSSL_memset(buf, 0, in_len);
  state[0] = AES_BLOCK_XOR(state[0], AES_BLOCK_LOAD(buf));
  state[4] = AES_BLOCK_XOR(state[4], AES_BLOCK_LOAD(buf + AES_BLOCK_LENGTH));
}

static int aead_aegis_128x2_openv_detached(
    const EVP_AEAD_CTX *ctx, bssl::Span<const CRYPTO_IOVEC> iovecs,
    bssl::Span<const uint8_t> nonce, bssl::Span<const uint8_t> in_tag,
    bssl::Span<const CRYPTO_IVEC> aadvecs) {
  const struct aead_aegis_128x2_ctx *aegis_ctx =
      (struct aead_aegis_128x2_ctx *)&ctx->state;

  if (nonce.size() > AEGIS_128X2_NONCE_LEN) {
    OPENSSL_PUT_ERROR(CIPHER, CIPHER_R_UNSUPPORTED_NONCE_SIZE);
    return 0;
  }
  if (in_tag.size() != ctx->tag_len) {
    OPENSSL_PUT_ERROR(CIPHER, CIPHER_R_BAD_DECRYPT);
    return 0;
  }

  size_t ad_len = bssl::iovec::TotalLength(aadvecs);
  size_t in_len = bssl::iovec::TotalLength(iovecs);

  alignas(64) aegis_128x2_state state;
  aegis_128x2_state_init(aegis_ctx->key, nonce.data(), nonce.size(), state);

  aegis_128x2_absorb_ad(aadvecs, state);

  bssl::iovec::ForEachBlockRange<RATE, /*WriteOut=*/true>(
      iovecs,
      [&](const uint8_t *in, uint8_t *out, size_t len) {
        while (len >= RATE) {
          aead_aegis_128x2_dec(out, in, state);
          in += RATE;
          out += RATE;
          len -= RATE;
        }
        return true;
      },
      [&](const uint8_t *in, uint8_t *out, size_t len) {
        while (len >= RATE) {
          aead_aegis_128x2_dec(out, in, state);
          in += RATE;
          out += RATE;
          len -= RATE;
        }
        if (len > 0) {
          aegis_128x2_dec_partial(out, in, len, state);
        }
        return true;
      });

  uint8_t tag[AEGIS_128X2_TAG_LEN];
  aead_aegis_128x2_tag(tag, ad_len, in_len, state);

  OPENSSL_memset(state, 0, sizeof state);

  if (CRYPTO_memcmp(tag, in_tag.data(), ctx->tag_len) != 0) {
    OPENSSL_PUT_ERROR(CIPHER, CIPHER_R_BAD_DECRYPT);
    return 0;
  }

  return 1;
}

static const EVP_AEAD aead_aegis_128x2 = {
    AEGIS_128X2_KEY_LEN,    // key length
    AEGIS_128X2_NONCE_LEN,  // nonce length
    AEGIS_128X2_TAG_LEN,    // overhead
    AEGIS_128X2_TAG_LEN,    // max tag length

    aead_aegis_128x2_init,
    nullptr,  // init_with_direction
    aead_aegis_128x2_cleanup,
    nullptr,  // openv
    aead_aegis_128x2_sealv,
    aead_aegis_128x2_openv_detached,
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
const EVP_AEAD *EVP_aead_aegis_128x2(void) {
  if (CRYPTO_is_AVX2_capable() && CRYPTO_is_VAES_capable()) {
    return &aead_aegis_128x2;
  }
  return NULL;
}
#elif defined(OPENSSL_AARCH64)
const EVP_AEAD *EVP_aead_aegis_128x2(void) {
  if (CRYPTO_is_NEON_capable()) {
    return &aead_aegis_128x2;
  }
  return NULL;
}
#else
const EVP_AEAD *EVP_aead_aegis_128x2(void) { return NULL; }
#endif
