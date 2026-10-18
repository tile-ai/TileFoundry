#pragma once
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

#define AMX_NOP_OP_IMM5(op, imm5)                                              \
  __asm("nop\nnop\nnop\n.word (0x201000 + (%0 << 5) + %1)"                     \
        :                                                                      \
        : "i"(op), "i"(imm5)                                                   \
        : "memory")

#define AMX_OP_GPR(op, gpr)                                                    \
  __asm(".word (0x201000 + (%0 << 5) + 0%1 - ((0%1 >> 4) * 6))"                \
        :                                                                      \
        : "i"(op), "r"((uint64_t)(gpr))                                        \
        : "memory")

#define AMX_LDX(gpr) AMX_OP_GPR(0, gpr)
#define AMX_LDY(gpr) AMX_OP_GPR(1, gpr)
#define AMX_STX(gpr) AMX_OP_GPR(2, gpr)
#define AMX_STY(gpr) AMX_OP_GPR(3, gpr)
#define AMX_LDZ(gpr) AMX_OP_GPR(4, gpr)
#define AMX_STZ(gpr) AMX_OP_GPR(5, gpr)
#define AMX_LDZI(gpr) AMX_OP_GPR(6, gpr)
#define AMX_STZI(gpr) AMX_OP_GPR(7, gpr)
#define AMX_EXTRX(gpr) AMX_OP_GPR(8, gpr)
#define AMX_EXTRY(gpr) AMX_OP_GPR(9, gpr)
#define AMX_FMA64(gpr) AMX_OP_GPR(10, gpr)
#define AMX_FMS64(gpr) AMX_OP_GPR(11, gpr)
#define AMX_FMA32(gpr) AMX_OP_GPR(12, gpr)
#define AMX_FMS32(gpr) AMX_OP_GPR(13, gpr)
#define AMX_MAC16(gpr) AMX_OP_GPR(14, gpr)
#define AMX_FMA16(gpr) AMX_OP_GPR(15, gpr)
#define AMX_FMS16(gpr) AMX_OP_GPR(16, gpr)
#define AMX_SET() AMX_NOP_OP_IMM5(17, 0)
#define AMX_CLR() AMX_NOP_OP_IMM5(17, 1)
#define AMX_VECINT(gpr) AMX_OP_GPR(18, gpr)
#define AMX_VECFP(gpr) AMX_OP_GPR(19, gpr)
#define AMX_MATINT(gpr) AMX_OP_GPR(20, gpr)
#define AMX_MATFP(gpr) AMX_OP_GPR(21, gpr)
#define AMX_GENLUT(gpr) AMX_OP_GPR(22, gpr)

#ifdef __cplusplus
}
#endif

template <class T> struct operand {
  inline constexpr uint64_t bind(void *ptr) const noexcept {
    return val | (((1ull << 57) - 1) & (uint64_t)ptr);
  }

  inline constexpr operator uint64_t() const noexcept { return val; }

  template <uint64_t OffSet, uint64_t Width>
  inline constexpr T set_field(const uint64_t value) const noexcept {
    constexpr uint64_t clr_mask = ~(((1ull << Width) - 1ull) << OffSet);
    uint64_t value_mask = (value << OffSet);
    return T{(val & clr_mask) | value_mask};
  }

  uint64_t val;
};

struct ldxy : operand<ldxy> {
  /**
   * @brief load multiple register
   *
   * factor = multiple ? 2 : 1
   *
   * factor2 = multiple_means_four ? 2 : 1
   *
   * so total load factor * factor2 * 64 byte
   */
  constexpr auto multiple() const { return set_field<62, 1>(1); }
  constexpr auto non_consecutive() const { return set_field<61, 1>(1); }
  constexpr auto multiple_four() const { return set_field<60, 1>(1); }
  /**
   * @brief choice reg id, 0~5
   */
  constexpr auto register_index(const uint64_t index) const {
    return set_field<56, 3>(index);
  }
};

struct stxy : operand<stxy> {
  /**
   * @brief store (multiple ? 2 : 1) * 64 bytes.
   */
  constexpr auto multiple() const { return set_field<62, 1>(1); }
  /**
   * @brief choice reg id, 0~5
   */
  constexpr auto register_index(const uint64_t index) const {
    return set_field<56, 3>(index);
  }
};

struct ldstz : operand<ldstz> {
  /**
   * @brief load or store (multiple ? 2 : 1) * register.
   */
  constexpr auto multiple() const { return set_field<62, 1>(1); }
  /**
   * @brief choice row id, 0 ~ 63
   */
  constexpr auto row_index(const uint64_t index) const {
    return set_field<56, 6>(index);
  }
};

struct ldstzi : operand<ldstzi> {
  /**
   * @brief interleave load one row to two row.
   *   right: x[0:16:2] => z[row,8:16], x[1:16:2] => z[row+1,8:16]
   *   left:  x[0:16:2] => z[row,0:8], x[1:16:2] => z[row+1,0:8]
   */
  constexpr auto right() const { return set_field<56, 1>(1); }
  /**
   * @brief choice row id, 0 ~ 63
   */
  constexpr auto row_index(const uint64_t index) const {
    return set_field<57, 5>(index);
  }
};

/**
 * @brief
 *  matrix mode:
 *  the x, y is one register(64 bytes).
 *    z[j][z_row][i] += x[i] * y[j]
 *  note 0 <= z_row < 64 / lane,  0 <= j < 64 / z_row.
 *  vector mode: z[_][i] += x[i] * y[i]
 */
template <class T> struct fma_base : operand<T> {
  constexpr auto vector_mode() const {
    return this->template set_field<63, 1>(1);
  }
  /**
   * @brief x_mode and x_value is combined options.
   * 0: Enable all lanes (x_value==0), or odd lanes only (x_value==1), or even
   * lanes only (x_value==2), or no lanes (x_value==anything else)
   *
   * 1: Only enable lane x_value
   *
   * 2: Only enable the first x_value lanes, or all lanes when x_value is zero
   *
   * 3: Only enable the last x_value lanes, or all lanes when x_value is zero
   */
  constexpr auto x_mode(uint64_t v) const {
    return this->template set_field<46, 2>(v);
  }
  constexpr auto x_value(uint64_t v) const {
    return this->template set_field<41, 5>(v);
  }
  /**
   * @brief same like x_mode and x_value
   */
  constexpr auto y_mode(uint64_t v) const {
    return this->template set_field<37, 2>(v);
  }
  /**
   * @brief Ignored in vector mode
   */
  constexpr auto y_value(uint64_t v) const {
    return this->template set_field<32, 5>(v);
  }
  constexpr auto skip_x() const { return this->template set_field<29, 1>(1); }
  constexpr auto skip_y() const { return this->template set_field<28, 1>(1); }
  constexpr auto skip_z() const { return this->template set_field<27, 1>(1); }

  /**
   * @brief 0 <= z_row < 64 / lane, eg.
   *  fma64 : 0 <= z_row < 8 (lane = 8)
   *  fma32 : 0 <= z_row < 4 (lane = 16)
   *  fma16 : 0 <= z_row < 2 (lane = 32)
   */
  constexpr auto z_row(uint64_t v) const {
    return this->template set_field<20, 6>(v);
  }
  /**
   * @brief x_offset in bytes
   */
  constexpr auto x_offset(uint64_t v) const {
    return this->template set_field<10, 9>(v);
  }
  /**
   * @brief y_offset in bytes
   */
  constexpr auto y_offset(uint64_t v) const {
    return this->template set_field<0, 9>(v);
  }
};

struct fma16 : fma_base<fma16> {
  constexpr auto f32_z() const { return set_field<62, 1>(1); }
};

struct fma32 : fma_base<fma32> {
  constexpr auto f16_x() const { return set_field<61, 1>(1); }
  constexpr auto f16_y() const { return set_field<60, 1>(1); }
};

struct fma64 : fma_base<fma64> {
  constexpr auto f16_x() const { return set_field<61, 1>(1); }
  constexpr auto f16_y() const { return set_field<60, 1>(1); }
};

/**
 * @brief fms is just doing mul and sub.
 */
template <class T> struct fms_base : operand<T> {
  constexpr auto vector_mode() const {
    return this->template set_field<63, 1>(1);
  }
  /**
   * @brief x_mode and x_value is combined options.
   * 0: Enable all lanes (x_value==0), or odd lanes only (x_value==1), or even
   * lanes only (x_value==2), or no lanes (x_value==anything else)
   *
   * 1: Only enable lane x_value
   *
   * 2: Only enable the first x_value lanes, or all lanes when x_value is zero
   *
   * 3: Only enable the last x_value lanes, or all lanes when x_value is zero
   */
  constexpr auto x_mode(uint64_t v) const {
    return this->template set_field<46, 2>(v);
  }
  constexpr auto x_value(uint64_t v) const {
    return this->template set_field<41, 5>(v);
  }
  /**
   * @brief same like x_mode and x_value
   */
  constexpr auto y_mode(uint64_t v) const {
    return this->template set_field<37, 2>(v);
  }
  /**
   * @brief Ignored in vector mode
   */
  constexpr auto y_value(uint64_t v) const {
    return this->template set_field<32, 5>(v);
  }
  constexpr auto skip_x() const { return this->template set_field<29, 1>(1); }
  constexpr auto skip_y() const { return this->template set_field<28, 1>(1); }
  constexpr auto skip_z() const { return this->template set_field<27, 1>(1); }

  /**
   * @brief 0 <= z_row < 64 / lane, eg.
   *  fma64 : 0 <= z_row < 8 (lane = 8)
   *  fma32 : 0 <= z_row < 4 (lane = 16)
   *  fma16 : 0 <= z_row < 2 (lane = 32)
   */
  constexpr auto z_row(uint64_t v) const {
    return this->template set_field<20, 6>(v);
  }
  /**
   * @brief x_offset in bytes
   */
  constexpr auto x_offset(uint64_t v) const {
    return this->template set_field<10, 9>(v);
  }
  /**
   * @brief y_offset in bytes
   */
  constexpr auto y_offset(uint64_t v) const {
    return this->template set_field<0, 9>(v);
  }
};

struct fms16 : fms_base<fms16> {
  constexpr auto f32_z() const { return set_field<62, 1>(1); }
};

struct fms32 : fms_base<fms32> {
  constexpr auto f16_x() const { return set_field<61, 1>(1); }
  constexpr auto f16_y() const { return set_field<60, 1>(1); }
};

struct fms64 : fms_base<fms64> {
  constexpr auto f16_x() const { return set_field<61, 1>(1); }
  constexpr auto f16_y() const { return set_field<60, 1>(1); }
};

enum matfp_alu_mode : uint64_t {
  /**
   * @brief z + x*y
   *
   */
  kfma = 0,
  /**
   * @brief z - x*y
   */
  kfms = 1,
  /**
   * @brief x <= 0 ? 0 : y
   */
  cond = 4,
};

enum matfp_dtype_t : uint64_t {
  /**
   * @brief xy: bf16 z: bf16 (one row from each two)	M2 only
   */
  bf16bf16 = 0,
  /**
   * @brief xy: bf16 z: f32 (all rows, interleaved pairs)		M2 only
   */
  bf16f32 = 1,
  /**
   * @brief xy: f16 z: f16 (one row from each two)	anything else
   */
  f16f16 = 2,
  /**
   * @brief xy: f16 z: f32 (all rows, interleaved pairs)
   */
  f16f32 = 3,
  /**
   * @brief xy: f32 z: f32 (one row from each four)
   */
  f32f32 = 4,
  /**
   * @brief xy: f64 z: f64 (one row from each eight)
   */
  f64f64 = 7,
};

enum matfp_enable_mode : uint64_t {
  /**
   * @brief config mode by enable value.
   *  0 : Enable all lanes
   *  1 : odd lanes only
   *  2 : even lanes only
   *  3 : enable all lanes but override the ALU operation to 0.0
   *  4 or 5 : enable all lanes but override their value to 0.0
   *  others : no lanes enabled
   */
  config = 0,
  // Only enable lane #N
  kindex = 1,
  // Only enable the first N lanes, or all lanes when N is zero
  take = 2,
  // Only enable the last N lanes, or all lanes when N is zero
  take_last = 3,
  // Only enable the first N lanes (no lanes when N is zero)
  take_or_non = 4,
  // Only enable the last N lanes (no lanes when N is zero)
  take_last_or_non = 5,
  // No lanes enabled
  disable = 6,
  // No lanes enabled
  // disable2 = 7,
};

enum shuffle_mode : uint64_t {
  /**
   * @brief no shuffle.
   *
   */
  s0 = 0,
  /**
   * @brief pack half, eg. f32[16] view as f32[2,8] then pack axis 0.
   */
  s1 = 0,
  /**
   * @brief pack quarter, eg. f32[16] view as f32[4,4] then pack axis 0.
   */
  s2 = 0,
  /**
   * @brief pack one eighth, eg. f32[16] view as f32[8,2] then pack axis 0.
   */
  s3 = 0,
};

/**
 * @brief matfp : z[j][i] ±= f(x[i], y[j])
 *  support shuffle x/y/z and indexed load.
 */
struct matfp : operand<matfp> {

  /**
   * @brief depend on the y_mode
   */
  constexpr auto y_value(uint64_t v) const { return set_field<57, 6>(v); }

  constexpr auto index_load() const { return set_field<53, 1>(1); }

  /**
   * @brief only enabled when index_load() == 1.
   */
  constexpr auto index_register(uint64_t v) const {
    return set_field<49, 3>(v);
  }
  /**
   * @brief indices is 4 bit, otherwise is 2 bit
   */
  constexpr auto index_4bit() const { return set_field<48, 1>(1); }

  /**
   * @brief load to y, otherwise load to x.
   */
  constexpr auto indexed_to_y() const { return set_field<47, 1>(1); }

  /**
   * @brief only enabled when index_load() == 0
   */
  constexpr auto alu_mode(matfp_alu_mode v) const {
    return set_field<47, 6>(v);
  }

  /**
   * @brief Lane width mode
   */
  constexpr auto dtype_mode(matfp_dtype_t v) const { return set_field<42, 4>(v); }
  constexpr auto x_mode(matfp_enable_mode v) const {
    return set_field<38, 3>(v);
  }
  constexpr auto x_value(uint64_t v) const { return set_field<32, 5>(v); }
  constexpr auto x_shuffle(shuffle_mode v) const { return set_field<29, 2>(v); }
  constexpr auto y_shuffle(shuffle_mode v) const { return set_field<27, 2>(v); }
  constexpr auto y_mode(matfp_enable_mode v) const {
    return set_field<23, 3>(v);
  }
  constexpr auto z_row(uint64_t v) const { return set_field<20, 3>(v); }
  constexpr auto x_offset(uint64_t v) const { return set_field<10, 9>(v); }
  constexpr auto y_offset(uint64_t v) const { return set_field<0, 9>(v); }
};

/**
 * @brief  Data type (DT)   Index type (IT)   Lane count (LC)   Direction   mode
 */
enum genlut_mode : uint64_t {
  /** @brief
   * f32                  u4                 16           Generate   `0`
   */
  g_f32_u4_16 = 0,
  /** @brief
   * bf16 or f16          u5                 32           Generate   `1`   M2
   * required for bf16. See bit 30
   */
  g_f16_u5_32,
  /** @brief
   * f64                  u4                 8            Generate   `2` High
   * bit of index always generated as `0`
   */
  g_f64_u4_8,
  /** @brief
   * i32                  u4                 16           Generate   `3`
   */
  g_i32_u4_16,
  /** @brief
   * i16                  u5                 32           Generate   `4`
   */
  g_i16_u5_32,
  /** @brief
   * u32                  u4                 16           Generate   `5`
   */
  g_u32_u4_16,
  /** @brief
   * u16                  u5                 32           Generate   `6`
   */
  g_u16_u5_32,
  /** @brief
   * any 32-bit           u2                 16            Lookup   `7`
   */
  lc_a32_u2_16,
  /** @brief
   * any 16-bit           u2                 32            Lookup   `8`
   */
  lc_a16_u2_32,
  /** @brief
   * any 8-bit           u2                 64            Lookup   `9`
   */
  lc_a8_u2_64,
  /** @brief
   * any 64-bit           u4                 8            Lookup   `10` High bit
   * of index ignored
   */
  lc_a64_u4_8,
  /** @brief
   * any 32-bit           u4                 16            Lookup   `11`
   */
  lc_a32_u4_16,
  /** @brief
   * any 16-bit           u4                 32            Lookup   `12`
   */
  lc_a16_u4_32,
  /** @brief
   * any 8-bit           u4                 64            Lookup   `13`
   */
  lc_a8_u4_64,
  /** @brief
   * any 16-bit           u5                 32            Lookup   `14`
   */
  lc_a16_u5_32,
  /** @brief
   * any 8-bit           u5                 64            Lookup   `15`
   */
  lc_a8_u5_64,
};

struct genlut : operand<genlut> {

  /**
   * @brief the table register index
   */
  constexpr auto register_table(uint64_t v) const {
    return set_field<60, 3>(v);
  }

  /**
   * @brief Table is from Y or from X
   *
   */
  constexpr auto table_from_y() const { return set_field<59, 1>(1); }

  /**
   * @brief Table is from Y or from X
   * @table mode table
   */
  constexpr auto mode(genlut_mode v) const { return set_field<53, 4>(v); }

  /**
   * @brief Data type is bf16 (1) or f16 (0), enabled when mode==1 and m2 only.
   */
  constexpr auto bf16() const { return set_field<30, 1>(1); }

  /**
   * @brief Destination is Z (1) or is X or Y (0), Ignored in generate modes;
   * they always write to X or Y.
   */
  constexpr auto dest_z() const { return set_field<26, 1>(1); }

  /**
   * @brief enabled when dest_z == 0, Destination is Y (1) or is X (0)
   */
  constexpr auto dest_y() const { return set_field<25, 1>(1); }

  /**
   * @brief enabled when dest_z ==1, Destination register index (hi)
   */
  constexpr auto register_hi(uint64_t v) const { return set_field<23, 3>(v); }

  /**
   * @brief Destination register index (lo)
   */
  constexpr auto register_lo(uint64_t v) const { return set_field<20, 3>(v); }

  /**
   * @brief Source is from Y (1) or from X (0)
   */
  constexpr auto src_from_y() const { return set_field<10, 1>(1); }

  /**
   * @brief Source offset (in bytes)
   */
  constexpr auto src_offset(uint64_t v) const { return set_field<0, 9>(v); }
};
