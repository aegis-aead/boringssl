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

#define AEGIS_128X4_KEY_LEN 16
#define AEGIS_128X4_NONCE_LEN 16
#define AEGIS_128X4_TAG_LEN 16

#define AES_BLOCK_LENGTH 64
#define RATE (2 * AES_BLOCK_LENGTH)

#ifdef OPENSSL_X86_64
#ifdef __clang__
#pragma clang attribute push(__attribute__((target("vaes,avx512f,avx512bw,avx512vl,avx512dq"))), \
                             apply_to = function)
#elif defined(__GNUC__)
#pragma GCC target("vaes,avx512f,avx512bw,avx512vl,avx512dq")
#endif

#include <immintrin.h>

typedef __m512i aes_block_t;

#define AES_BLOCK_XOR(A, B) _mm512_xor_si512((A), (B))
#define AES_BLOCK_AND(A, B) _mm512_and_si512((A), (B))
#define AES_BLOCK_LOAD(A) _mm512_loadu_si512((const void *)(A))
#define AES_BLOCK_LOAD_64x2(A, B) \
  _mm512_broadcast_i64x2(_mm_set_epi64x((A), (B)))
#define AES_BLOCK_STORE(A, B) _mm512_storeu_si512((void *)(A), (B))
#define AES_ENC(A, B) _mm512_aesenc_epi128((A), (B))

struct aead_aegis_128x4_ctx {
  uint8_t key[AEGIS_128X4_KEY_LEN];
};

typedef aes_block_t aegis_128x4_state[8];

static_assert(sizeof(((EVP_AEAD_CTX *)NULL)->state) >=
                  sizeof(struct aead_aegis_128x4_ctx),
              "AEAD state is too small");
static_assert(alignof(union evp_aead_ctx_st_state) >=
                  alignof(struct aead_aegis_128x4_ctx),
              "AEAD state has insufficient alignment");

static int aead_aegis_128x4_init(EVP_AEAD_CTX *ctx, const uint8_t *key,
                                 size_t key_len, size_t tag_len) {
  if (key_len != AEGIS_128X4_KEY_LEN) {
    OPENSSL_PUT_ERROR(CIPHER, CIPHER_R_BAD_KEY_LENGTH);
    return 0;
  }
  if (tag_len == EVP_AEAD_DEFAULT_TAG_LENGTH) {
    tag_len = AEGIS_128X4_TAG_LEN;
  }
  if (tag_len != AEGIS_128X4_TAG_LEN) {
    OPENSSL_PUT_ERROR(CIPHER, CIPHER_R_TAG_TOO_LARGE);
    return 0;
  }

  struct aead_aegis_128x4_ctx *aegis_ctx =
      (struct aead_aegis_128x4_ctx *)&ctx->state;
  OPENSSL_memcpy(aegis_ctx->key, key, key_len);
  ctx->tag_len = tag_len;

  return 1;
}

static void aead_aegis_128x4_cleanup(EVP_AEAD_CTX *ctx) {}

static inline void aegis_128x4_state_update(aes_block_t *const state,
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

static void aegis_128x4_state_init(const uint8_t *key, const uint8_t *nonce,
                                   size_t nonce_len, aes_block_t *const state) {
  static const uint8_t c0_[] = {
      0xdb, 0x3d, 0x18, 0x55, 0x6d, 0xc2, 0x2f, 0xf1, 0x20, 0x11, 0x31,
      0x42, 0x73, 0xb5, 0x28, 0xdd, 0xdb, 0x3d, 0x18, 0x55, 0x6d, 0xc2,
      0x2f, 0xf1, 0x20, 0x11, 0x31, 0x42, 0x73, 0xb5, 0x28, 0xdd, 0xdb,
      0x3d, 0x18, 0x55, 0x6d, 0xc2, 0x2f, 0xf1, 0x20, 0x11, 0x31, 0x42,
      0x73, 0xb5, 0x28, 0xdd, 0xdb, 0x3d, 0x18, 0x55, 0x6d, 0xc2, 0x2f,
      0xf1, 0x20, 0x11, 0x31, 0x42, 0x73, 0xb5, 0x28, 0xdd};
  static const uint8_t c1_[] = {
      0x00, 0x01, 0x01, 0x02, 0x03, 0x05, 0x08, 0x0d, 0x15, 0x22, 0x37,
      0x59, 0x90, 0xe9, 0x79, 0x62, 0x00, 0x01, 0x01, 0x02, 0x03, 0x05,
      0x08, 0x0d, 0x15, 0x22, 0x37, 0x59, 0x90, 0xe9, 0x79, 0x62, 0x00,
      0x01, 0x01, 0x02, 0x03, 0x05, 0x08, 0x0d, 0x15, 0x22, 0x37, 0x59,
      0x90, 0xe9, 0x79, 0x62, 0x00, 0x01, 0x01, 0x02, 0x03, 0x05, 0x08,
      0x0d, 0x15, 0x22, 0x37, 0x59, 0x90, 0xe9, 0x79, 0x62,
  };
  const aes_block_t c0 = AES_BLOCK_LOAD(c0_);
  const aes_block_t c1 = AES_BLOCK_LOAD(c1_);
  uint8_t context_bytes[AES_BLOCK_LENGTH];
  aes_block_t context;

  uint8_t padded_nonce[AEGIS_128X4_NONCE_LEN] = {0};
  assert(nonce_len <= sizeof padded_nonce);
  OPENSSL_memcpy(padded_nonce, nonce, nonce_len);

  uint8_t key_x[AES_BLOCK_LENGTH];
  uint8_t nonce_x[AES_BLOCK_LENGTH];
  memcpy(key_x, key, AEGIS_128X4_KEY_LEN);
  memcpy(key_x + AEGIS_128X4_KEY_LEN, key, AEGIS_128X4_KEY_LEN);
  memcpy(key_x + 2 * AEGIS_128X4_KEY_LEN, key, AEGIS_128X4_KEY_LEN);
  memcpy(key_x + 3 * AEGIS_128X4_KEY_LEN, key, AEGIS_128X4_KEY_LEN);
  memcpy(nonce_x, padded_nonce, AEGIS_128X4_NONCE_LEN);
  memcpy(nonce_x + AEGIS_128X4_NONCE_LEN, padded_nonce, AEGIS_128X4_NONCE_LEN);
  memcpy(nonce_x + 2 * AEGIS_128X4_NONCE_LEN, padded_nonce,
         AEGIS_128X4_NONCE_LEN);
  memcpy(nonce_x + 3 * AEGIS_128X4_NONCE_LEN, padded_nonce,
         AEGIS_128X4_NONCE_LEN);

  aes_block_t k = AES_BLOCK_LOAD(key_x);
  aes_block_t n = AES_BLOCK_LOAD(nonce_x);

  memset(context_bytes, 0, sizeof context_bytes);
  context_bytes[0 * 16] = 0x00;
  context_bytes[0 * 16 + 1] = 0x03;
  context_bytes[1 * 16] = 0x01;
  context_bytes[1 * 16 + 1] = 0x03;
  context_bytes[2 * 16] = 0x02;
  context_bytes[2 * 16 + 1] = 0x03;
  context_bytes[3 * 16] = 0x03;
  context_bytes[3 * 16 + 1] = 0x03;
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
    aegis_128x4_state_update(state, n, k);
  }
}

static void aead_aegis_128x4_tag(uint8_t *tag, size_t adlen, size_t mlen,
                                 aes_block_t *const state) {
  aes_block_t tmp;

  tmp = AES_BLOCK_LOAD_64x2((uint64_t)mlen << 3, (uint64_t)adlen << 3);
  tmp = AES_BLOCK_XOR(tmp, state[2]);

  for (int i = 0; i < 7; i++) {
    aegis_128x4_state_update(state, tmp, tmp);
  }

  uint8_t mac_multi[AES_BLOCK_LENGTH];
  tmp = AES_BLOCK_XOR(state[6], AES_BLOCK_XOR(state[5], state[4]));
  tmp = AES_BLOCK_XOR(tmp, AES_BLOCK_XOR(state[3], state[2]));
  tmp = AES_BLOCK_XOR(tmp, AES_BLOCK_XOR(state[1], state[0]));
  AES_BLOCK_STORE(mac_multi, tmp);
  for (size_t i = 0; i < 16; i++) {
    tag[i] = mac_multi[i] ^ mac_multi[1 * 16 + i] ^ mac_multi[2 * 16 + i] ^
             mac_multi[3 * 16 + i];
  }
}

static void aead_aegis_128x4_enc(uint8_t *const dst, const uint8_t *const src,
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

  aegis_128x4_state_update(state, msg0, msg1);
}

static void aead_aegis_128x4_dec(uint8_t *const dst, const uint8_t *const src,
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

  aegis_128x4_state_update(state, msg0, msg1);
}

static void aegis_128x4_absorb_ad(bssl::Span<const CRYPTO_IVEC> aadvecs,
                                  aegis_128x4_state state) {
  alignas(AES_BLOCK_LENGTH) uint8_t buf[RATE];
  bssl::iovec::ForEachBlockRange<RATE>(
      aadvecs,
      [&](const uint8_t *in, size_t len) {
        while (len >= RATE) {
          aead_aegis_128x4_enc(buf, in, state);
          in += RATE;
          len -= RATE;
        }
        return true;
      },
      [&](const uint8_t *in, size_t len) {
        while (len >= RATE) {
          aead_aegis_128x4_enc(buf, in, state);
          in += RATE;
          len -= RATE;
        }
        if (len > 0) {
          OPENSSL_memset(buf, 0, RATE);
          OPENSSL_memcpy(buf, in, len);
          aead_aegis_128x4_enc(buf, buf, state);
        }
        return true;
      });
}

static int aead_aegis_128x4_sealv(const EVP_AEAD_CTX *ctx,
                                  bssl::Span<const CRYPTO_IOVEC> iovecs,
                                  bssl::Span<uint8_t> out_tag,
                                  size_t *out_tag_len,
                                  bssl::Span<const uint8_t> nonce,
                                  bssl::Span<const CRYPTO_IVEC> aadvecs) {
  const struct aead_aegis_128x4_ctx *aegis_ctx =
      (struct aead_aegis_128x4_ctx *)&ctx->state;

  if (out_tag.size() < ctx->tag_len) {
    OPENSSL_PUT_ERROR(CIPHER, CIPHER_R_BUFFER_TOO_SMALL);
    return 0;
  }
  if (nonce.size() > AEGIS_128X4_NONCE_LEN) {
    OPENSSL_PUT_ERROR(CIPHER, CIPHER_R_UNSUPPORTED_NONCE_SIZE);
    return 0;
  }

  size_t ad_len = bssl::iovec::TotalLength(aadvecs);
  size_t in_len = bssl::iovec::TotalLength(iovecs);

  alignas(64) aegis_128x4_state state;
  aegis_128x4_state_init(aegis_ctx->key, nonce.data(), nonce.size(), state);

  aegis_128x4_absorb_ad(aadvecs, state);

  bssl::iovec::ForEachBlockRange<RATE, /*WriteOut=*/true>(
      iovecs,
      [&](const uint8_t *in, uint8_t *out, size_t len) {
        while (len >= RATE) {
          aead_aegis_128x4_enc(out, in, state);
          in += RATE;
          out += RATE;
          len -= RATE;
        }
        return true;
      },
      [&](const uint8_t *in, uint8_t *out, size_t len) {
        while (len >= RATE) {
          aead_aegis_128x4_enc(out, in, state);
          in += RATE;
          out += RATE;
          len -= RATE;
        }
        if (len > 0) {
          alignas(AES_BLOCK_LENGTH) uint8_t buf[RATE];
          OPENSSL_memset(buf, 0, RATE);
          OPENSSL_memcpy(buf, in, len);
          aead_aegis_128x4_enc(buf, buf, state);
          OPENSSL_memcpy(out, buf, len);
        }
        return true;
      });

  aead_aegis_128x4_tag(out_tag.data(), ad_len, in_len, state);
  *out_tag_len = ctx->tag_len;

  OPENSSL_memset(state, 0, sizeof state);

  return 1;
}

static void aegis_128x4_dec_partial(uint8_t *out, const uint8_t *in,
                                    size_t in_len, aegis_128x4_state state) {
  alignas(AES_BLOCK_LENGTH) uint8_t buf[RATE];
  OPENSSL_memset(buf, 0, RATE);
  OPENSSL_memcpy(buf, in, in_len);
  aead_aegis_128x4_dec(buf, buf, state);
  OPENSSL_memcpy(out, buf, in_len);
  OPENSSL_memset(buf, 0, in_len);
  state[0] = AES_BLOCK_XOR(state[0], AES_BLOCK_LOAD(buf));
  state[4] = AES_BLOCK_XOR(state[4], AES_BLOCK_LOAD(buf + AES_BLOCK_LENGTH));
}

static int aead_aegis_128x4_openv_detached(
    const EVP_AEAD_CTX *ctx, bssl::Span<const CRYPTO_IOVEC> iovecs,
    bssl::Span<const uint8_t> nonce, bssl::Span<const uint8_t> in_tag,
    bssl::Span<const CRYPTO_IVEC> aadvecs) {
  const struct aead_aegis_128x4_ctx *aegis_ctx =
      (struct aead_aegis_128x4_ctx *)&ctx->state;

  if (nonce.size() > AEGIS_128X4_NONCE_LEN) {
    OPENSSL_PUT_ERROR(CIPHER, CIPHER_R_UNSUPPORTED_NONCE_SIZE);
    return 0;
  }
  if (in_tag.size() != ctx->tag_len) {
    OPENSSL_PUT_ERROR(CIPHER, CIPHER_R_BAD_DECRYPT);
    return 0;
  }

  size_t ad_len = bssl::iovec::TotalLength(aadvecs);
  size_t in_len = bssl::iovec::TotalLength(iovecs);

  alignas(64) aegis_128x4_state state;
  aegis_128x4_state_init(aegis_ctx->key, nonce.data(), nonce.size(), state);

  aegis_128x4_absorb_ad(aadvecs, state);

  bssl::iovec::ForEachBlockRange<RATE, /*WriteOut=*/true>(
      iovecs,
      [&](const uint8_t *in, uint8_t *out, size_t len) {
        while (len >= RATE) {
          aead_aegis_128x4_dec(out, in, state);
          in += RATE;
          out += RATE;
          len -= RATE;
        }
        return true;
      },
      [&](const uint8_t *in, uint8_t *out, size_t len) {
        while (len >= RATE) {
          aead_aegis_128x4_dec(out, in, state);
          in += RATE;
          out += RATE;
          len -= RATE;
        }
        if (len > 0) {
          aegis_128x4_dec_partial(out, in, len, state);
        }
        return true;
      });

  uint8_t tag[AEGIS_128X4_TAG_LEN];
  aead_aegis_128x4_tag(tag, ad_len, in_len, state);

  OPENSSL_memset(state, 0, sizeof state);

  if (CRYPTO_memcmp(tag, in_tag.data(), ctx->tag_len) != 0) {
    OPENSSL_PUT_ERROR(CIPHER, CIPHER_R_BAD_DECRYPT);
    return 0;
  }

  return 1;
}

static const EVP_AEAD aead_aegis_128x4 = {
    AEGIS_128X4_KEY_LEN,    // key length
    AEGIS_128X4_NONCE_LEN,  // nonce length
    AEGIS_128X4_TAG_LEN,    // overhead
    AEGIS_128X4_TAG_LEN,    // max tag length

    aead_aegis_128x4_init,
    nullptr,  // init_with_direction
    aead_aegis_128x4_cleanup,
    nullptr,  // openv
    aead_aegis_128x4_sealv,
    aead_aegis_128x4_openv_detached,
    nullptr,  // get_iv
    nullptr,  // tag_len
};

#ifdef __clang__
#pragma clang attribute pop
#endif

const EVP_AEAD *EVP_aead_aegis_128x4(void) {
  if (CRYPTO_is_AVX512BW_capable() && CRYPTO_is_AVX512VL_capable() &&
      CRYPTO_is_VAES_capable() && !CRYPTO_cpu_avoid_zmm_registers()) {
    return &aead_aegis_128x4;
  }
  return NULL;
}

#else  // OPENSSL_X86_64

const EVP_AEAD *EVP_aead_aegis_128x4(void) { return NULL; }

#endif  // OPENSSL_X86_64
