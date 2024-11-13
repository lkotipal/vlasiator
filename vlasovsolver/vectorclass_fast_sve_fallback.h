#ifndef _fast_sve
#define _fast_sve(...)

#ifdef VEC8F_SVE_SIPEARL
typedef svfloat32_t svfixed_float32x8_t __attribute__((arm_sve_vector_bits(256)));
typedef svint32_t svfixed_int32x8_t __attribute__((arm_sve_vector_bits(256)));
typedef svbool_t svfixed_predicatex8_t __attribute__((arm_sve_vector_bits(256)));
class Vec8bf;
class Vec8f;
class Vec8i;
#endif

#ifdef VEC4F_SVE_SIPEARL
typedef svfloat32_t svfixed_float32x4_t __attribute__((arm_sve_vector_bits(128)));
typedef svint32_t svfixed_int32x4_t __attribute__((arm_sve_vector_bits(128)));
typedef svbool_t svfixed_predicatex4_t __attribute__((arm_sve_vector_bits(128)));
class Vec4i;
#endif

#include <iostream>




// Prefeching is not implemented
#ifndef _mm_prefetch
#define _mm_prefetch(...)
#endif

#ifndef _define_subnormals
#define _define_subnormals(...)

/*static intptr_t  getFpStatusRegister(){
   intptr_t fpsr = 0;
   asm volatile("mrs %0, fpcr" : "=r" (fpsr));
   //asm volatile("vmrs %0, fpscr" : "=r" (fpsr));

   return fpsr;
}

static void setFpStatusRegister (intptr_t fpsr){
   asm volatile("msr fpcr, %0" : : "ri" (fpsr));
   //asm volatile("vmsr fpscr, %0" : : "ri" (fpsr));
}

static void no_subnormals()
{
   bool aux= true;
   intptr_t mask = (1 << 24 );  // FZ
   setFpStatusRegister ((getFpStatusRegister() & (~mask)) | (aux ? mask : 0));

}*/

static void no_subnormals()
{
}
#endif

// #################################################################

// Vec8i implementation

#ifdef VEC8F_SVE_SIPEARL
class Vec8i
{
public:
   svfixed_int32x8_t data;
   // Default constructor:
   Vec8i()
   {
   }
   Vec8i(svfixed_int32x8_t x)
   {
      data = x;
   }
   // Constructor to broadcast the same value into all elements:
   Vec8i(int f)
   {
      data = svdup_n_s32(f);
   }
   // Replicate 8 values across v.
   Vec8i(int a, int b, int c, int d, int e, int f, int g, int h)
   {
      int val[8] = {a, b, c, d, e, f, g, h};
      data = svld1(svptrue_b32(), val);
   }
   Vec8i(Vec8i const &x)
   {
      data = x.data;
   }
   // Member function to load from array (unaligned)
   Vec8i &load(int const *p)
   {
      data = svld1(svptrue_b32(), p);
      return *this;
   }

   // Member function to load from array, aligned by 32
   Vec8i &load_a(int const *p)
   {
      return this->load(p);
   }

   Vec8i &insert(int i, int const &x)
   {
      int aux[8];
      svst1_s32(svptrue_b32(), aux, data);
      aux[i] = x;
      data = svld1(svptrue_b32(), aux);
      return *this;
   }

   // Member function to store into array (unaligned)
   void store(int *x) const
   {
      svst1_s32(svptrue_b32(), x, data);
   }

   // Member function to store into array, aligned by 32
   void store_a(int *p) const
   {
      this->store(p);
   }

   Vec8i &operator=(Vec8i const &r)
   {
      data = r.data;
      return *this;
   }
   int extract(int index) const
   {
      return svclastb_n_s32(svwhilele_b32_s32(0, index), 0, data);
   }

   int operator[](int index) const
   {
      return svclastb_n_s32(svwhilele_b32_s32(0, index), 0, data);
   }

   Vec8i operator++(int)
   {
      data = svadd_n_s32_x(svptrue_b32(), this->data, 1);
      return *this;
   }
};

class Vec8f
{
public:
   svfixed_float32x8_t data;
   // Default constructor:
   Vec8f()
   {
   }
   Vec8f(svfixed_float32x8_t x)
   {
      data = x;
   }
   // Constructor to broadcast the same value into all elements:
   Vec8f(float f)
   {
      data = svdup_n_f32(f);
   }
   // Replicate 8 values across v.
   Vec8f(float a, float b, float c, float d, float e, float f, float g, float h)
   {
      float val[8] = {a, b, c, d, e, f, g, h};
      data = svld1(svptrue_b32(), &val[0]);
   }
   // Copy vector v.
   Vec8f(Vec8f const &x)
   {
      data = x.data;
   }
   // Member function to load from array (unaligned)
   Vec8f &load(float const *p)
   {
      data = svld1(svptrue_b32(), p);
      return *this;
   }

   // Member function to load from array, aligned by 32
   Vec8f &load_a(float const *p)
   {
      return this->load(p);
   }

   Vec8f &insert(int i, float const &x)
   {
      float aux[8];
      svst1_f32(svptrue_b32(), aux, data);
      aux[i] = x;
      data = svld1(svptrue_b32(), aux);
      return *this;
   }

   // Member function to store into array (unaligned)
   void store(float *x) const
   {
      svst1_f32(svptrue_b32(), x, data);
   }

   // Member function to store into array, aligned by 32
   void store_a(float *p) const
   {
      this->store(p);
   }

   Vec8f &operator=(Vec8f const &r)
   {
      data = r.data;
      return *this;
   }

   float extract(int index) const
   {
      return svclastb_n_f32(svwhilele_b32_u32(0, index), 0.0f, data);
   }

   float operator[](int index) const
   {
      return svclastb_n_f32(svwhilele_b32_u32(0, index), 0.0f, data);
   }

   Vec8f operator++(int)
   {
      data = svadd_n_f32_x(svptrue_b32(), this->data, 1.0f);
      return *this;
   }
};

// #################################################################

// Vec8b implementation
class Vec8bf
{
public:
   svfixed_predicatex8_t data;
   // Default constructor:
   Vec8bf()
   {
   }
   Vec8bf(svfixed_predicatex8_t x)
   {
      data = x;
   }

   Vec8bf(bool x0, bool x1, bool x2, bool x3, bool x4, bool x5, bool x6, bool x7)
   {
      Vec8f veca = {static_cast<float>(x0), static_cast<float>(x1), static_cast<float>(x2), static_cast<float>(x3), static_cast<float>(x4), static_cast<float>(x5), static_cast<float>(x6), static_cast<float>(x7)};
      data = svcmpne_f32(svptrue_b32(), veca.data, svdup_n_f32(0.0f));
   }

   bool operator[](int index) const
   {
      Vec8f result = svsel_f32(data, svdup_n_f32(1.0f), svdup_n_f32(0.0f));
      return result[index] == 0.0f ? false : true;
   }
};

static inline Vec8f abs(const Vec8f &l)
{
   return Vec8f(svabs_f32_x(svptrue_b32(), l.data));
}

static inline Vec8f sqrt(const Vec8f &l)
{
   return Vec8f(svsqrt_f32_x(svptrue_b32(), l.data));
}

static inline Vec8f operator+(Vec8f const &a, Vec8f const &b)
{
   return Vec8f(svadd_f32_x(svptrue_b32(), a.data, b.data));
}

static inline Vec8f operator+(Vec8f const &a, float b)
{
   return Vec8f(svadd_n_f32_x(svptrue_b32(), a.data, b));
}

static inline Vec8f operator+(float a, Vec8f const &b)
{
   return Vec8f(svadd_n_f32_x(svptrue_b32(), b.data, a));
}

static inline Vec8f operator-(const Vec8f &r)
{
   return Vec8f(svneg_f32_x(svptrue_b32(), r.data));
}

static inline Vec8f operator-(Vec8f const &a, Vec8f const &b)
{
   return Vec8f(svsub_f32_x(svptrue_b32(), a.data, b.data));
}

static inline Vec8f operator-(Vec8f const &a, float b)
{
   return Vec8f(svsub_n_f32_x(svptrue_b32(), a.data, b));
}

static inline Vec8f operator-(float a, Vec8f const &b)
{
   return Vec8f(svsub_f32_x(svptrue_b32(), svdup_n_f32(a), b.data));
}

static inline Vec8f operator*(Vec8f const &a, Vec8f const &b)
{
   return Vec8f(svmul_f32_x(svptrue_b32(), a.data, b.data));
}

static inline Vec8f operator*(Vec8f const &a, float b)
{
   return Vec8f(svmul_n_f32_x(svptrue_b32(), a.data, b));
}

static inline Vec8f operator*(float a, Vec8f const &b)
{
   return Vec8f(svmul_n_f32_x(svptrue_b32(), b.data, a));
}

static inline Vec8f operator/(const Vec8f &l, const Vec8f &r)
{
   return Vec8f(svdiv_f32_x(svptrue_b32(), l.data, r.data));
}

static inline Vec8f operator/(const Vec8f &l, const float &r)
{
   return Vec8f(svdiv_n_f32_x(svptrue_b32(), l.data, r));
}

static inline Vec8f operator/(const float &r, const Vec8f &l)
{
   return Vec8f(svdiv_f32_x(svptrue_b32(), svdup_n_f32(r), l.data));
}

static inline Vec8f &operator+=(Vec8f &l, const Vec8f &r)
{
   l.data = svadd_f32_x(svptrue_b32(), l.data, r.data);
   return l;
}

static inline Vec8f &operator+=(Vec8f &l, const float &r)
{
   l.data = svadd_f32_x(svptrue_b32(), l.data, svdup_n_f32(r));
   return l;
}

static inline Vec8f &operator-=(Vec8f &l, const Vec8f &r)
{
   l = l - r;
   return l;
}

static inline Vec8f &operator-=(Vec8f &l, const float &r)
{
   l = l - r;
   return l;
}

// operator ||

// operator &&

static inline Vec8bf operator==(const Vec8f &l, const Vec8f &r)
{
   return Vec8bf(svcmpeq_f32(svptrue_b32(), l.data, r.data));
}

static inline Vec8bf operator!=(const Vec8f &l, const Vec8f &r)
{
   return Vec8bf(svcmpne_f32(svptrue_b32(), l.data, r.data));
}

static inline Vec8bf operator!=(const Vec8f &l, const float &r)
{
   return Vec8bf(svcmpne_n_f32(svptrue_b32(), l.data, r));
}

static inline Vec8bf operator>(const Vec8f &l, const Vec8f &r)
{
   return Vec8bf(svcmpgt_f32(svptrue_b32(), l.data, r.data));
}

static inline Vec8bf operator>(const Vec8f &l, const float r)
{
   return Vec8bf(svcmpgt_n_f32(svptrue_b32(), l.data, r));
}

static inline Vec8bf operator>(const float l, const Vec8f &r)
{
   return Vec8bf(svcmplt_n_f32(svptrue_b32(), r.data, l));
}

static inline Vec8bf operator>=(const Vec8f &l, const Vec8f &r)
{
   return Vec8bf(svcmpge_f32(svptrue_b32(), l.data, r.data));
}

static inline Vec8bf operator>=(const Vec8f &l, const float r)
{
   return Vec8bf(svcmple_n_f32(svptrue_b32(), l.data, r));
}

static inline Vec8bf operator>=(const float l, const Vec8f &r)
{
   return Vec8bf(svcmpge_n_f32(svptrue_b32(), r.data, l));
}

static inline Vec8bf operator<(const Vec8f &l, const float &r)
{
   return Vec8bf(svcmplt_n_f32(svptrue_b32(), l.data, r));
}

static inline Vec8bf operator<(const float l, const Vec8f &r)
{
   return Vec8bf(svcmpgt_n_f32(svptrue_b32(), r.data, l));
}
static inline Vec8bf operator<(const Vec8f &l, const Vec8f &r)
{
   return Vec8bf(svcmplt_f32(svptrue_b32(), l.data, r.data));
}

static inline Vec8bf operator<=(const Vec8f &l, const Vec8f &r)
{
   return Vec8bf(svcmple_f32(svptrue_b32(), l.data, r.data));
}

static inline Vec8bf operator<=(const float l, const Vec8f &r)
{
   return Vec8bf(svcmpge_n_f32(svptrue_b32(), r.data, l));
}

static inline Vec8f min(const Vec8f &l, const Vec8f &r)
{
   return Vec8f(svmin_f32_x(svptrue_b32(), l.data, r.data));
}

static inline Vec8f min(float const &l, Vec8f const &r)
{
   return Vec8f(svmin_n_f32_x(svptrue_b32(), r.data, l));
}

static inline Vec8f min(Vec8f const &r, float const &l)
{
   return Vec8f(svmin_n_f32_x(svptrue_b32(), r.data, l));
}

static inline Vec8f max(const Vec8f &l, const Vec8f &r)
{
   return Vec8f(svmax_f32_x(svptrue_b32(), l.data, r.data));
}

static inline Vec8f max(float const &l, Vec8f const &r)
{
   return Vec8f(svmax_n_f32_x(svptrue_b32(), r.data, l));
}

static inline Vec8f max(Vec8f const &r, float const &l)
{
   return Vec8f(svmax_n_f32_x(svptrue_b32(), r.data, l));
}

static inline Vec8f select(Vec8bf const &a, Vec8f const &b, Vec8f const &c)
{
   return Vec8f(svsel_f32(a.data, b.data, c.data));
}

static inline Vec8f select(Vec8bf const &a, float const &b, Vec8f const &c)
{

   return Vec8f(svsel_f32(a.data, svdup_n_f32(b), c.data));
}

static inline Vec8f select(Vec8bf const &a, Vec8f const &b, float const &c)
{
   return Vec8f(svsel_f32(a.data, b.data, svdup_n_f32(c)));
}

static inline Vec8f select(Vec8bf const &a, float const &b, float const &c)
{
   return Vec8f(svsel_f32(a.data, svdup_n_f32(b), svdup_n_f32(c)));
}

// c + a * b
static inline Vec8f fused_add_multiply(Vec8f a, Vec8f b, Vec8f c){
    return Vec8f(svmad_f32_z(svptrue_b32(),a.data,b.data,c.data));
}

// c - a * b
static inline Vec8f fused_substract_multiply(Vec8f a, Vec8f b, Vec8f c){
    return Vec8f(svmad_f32_z(svptrue_b32(),a.data,b.data,svneg_f32_z(svptrue_b32(),c.data)));
}

static inline Vec8i abs(const Vec8i &l)
{
   return Vec8i(svabs_s32_x(svptrue_b32(), l.data));
}

static inline Vec8i operator+(Vec8i const &a, Vec8i const &b)
{
   return Vec8i(svadd_s32_x(svptrue_b32(), a.data, b.data));
}

// vector operator + : add vector and scalar
static inline Vec8i operator+(Vec8i const &a, int b)
{
   return Vec8i(svadd_n_s32_x(svptrue_b32(), a.data, b));
}

static inline Vec8i operator+(int a, Vec8i const &b)
{
   return Vec8i(svadd_n_s32_x(svptrue_b32(), b.data, a));
}

static inline Vec8i operator-(const Vec8i &r)
{
   return Vec8i(svneg_s32_x(svptrue_b32(), r.data));
}

static inline Vec8i operator-(Vec8i const &a, Vec8i const &b)
{
   return Vec8i(svsub_s32_x(svptrue_b32(), a.data, b.data));
}

static inline Vec8i operator-(Vec8i const &a, int b)
{
   return Vec8i(svsub_n_s32_x(svptrue_b32(), a.data, b));
}

static inline Vec8i operator-(int a, Vec8i const &b)
{
   return Vec8i(svsub_s32_x(svptrue_b32(), (svptrue_b32(), svdup_n_s32(a)), b.data));
}

static inline Vec8i operator*(Vec8i const &a, Vec8i const &b)
{
   return Vec8i(svmul_s32_x(svptrue_b32(), a.data, b.data));
}

static inline Vec8i operator*(Vec8i const &a, int b)
{
   return Vec8i(svmul_n_s32_x(svptrue_b32(), a.data, b));
}

static inline Vec8i operator*(int a, Vec8i const &b)
{
   return Vec8i(svmul_n_s32_x(svptrue_b32(), b.data, a));
}

static inline Vec8i operator/(const Vec8i &l, const Vec8i &r)
{
   return Vec8i(svdiv_s32_x(svptrue_b32(), l.data, r.data));
}

static inline Vec8i operator/(const Vec8i &l, const int &r)
{
   return Vec8i(svdiv_n_s32_x(svptrue_b32(), l.data, r));
}

static inline Vec8i operator/(const int &r, const Vec8i &l)
{
   return Vec8i(svdiv_s32_x(svptrue_b32(), (svptrue_b32(), svdup_n_s32(r)), l.data));
}

static inline Vec8i &operator+=(Vec8i &l, const Vec8i &r)
{
   l = l + r;
   return l;
}

static inline Vec8i &operator+=(Vec8i &l, const int &r)
{
   l = l + r;
   return l;
}

static inline Vec8i &operator-=(Vec8i &l, const Vec8i &r)
{
   l = l - r;
   return l;
}

static inline Vec8i &operator-=(Vec8i &l, const int &r)
{
   l = l - r;
   return l;
}

// operator ||

// operator &&

static inline Vec8bf operator==(const Vec8i &l, const Vec8i &r)
{
   return Vec8bf(svcmpeq_s32(svptrue_b32(), l.data, r.data));
}

static inline Vec8bf operator!=(const Vec8i &l, const Vec8i &r)
{
   return Vec8bf(svcmpne_s32(svptrue_b32(), l.data, r.data));
}

static inline Vec8bf operator!=(const Vec8i &l, const int &r)
{
   return Vec8bf(svcmpne_n_s32(svptrue_b32(), l.data, r));
}

static inline Vec8bf operator>(const Vec8i &l, const Vec8i &r)
{
   return Vec8bf(svcmpgt_s32(svptrue_b32(), l.data, r.data));
}

static inline Vec8bf operator>(const Vec8i &l, const int r)
{
   return Vec8bf(svcmpgt_n_s32(svptrue_b32(), l.data, r));
}

static inline Vec8bf operator>(const int l, const Vec8i &r)
{
   return Vec8bf(svcmplt_n_s32(svptrue_b32(), r.data, l));
}

static inline Vec8bf operator>=(const Vec8i &l, const Vec8i &r)
{
   return Vec8bf(svcmpge_s32(svptrue_b32(), l.data, r.data));
}

static inline Vec8bf operator>=(const Vec8i &l, const int r)
{
   return Vec8bf(svcmple_n_s32(svptrue_b32(), l.data, r));
}

static inline Vec8bf operator>=(const int l, const Vec8i &r)
{
   return Vec8bf(svcmpge_n_s32(svptrue_b32(), r.data, l));
}

static inline Vec8bf operator<(const Vec8i &l, const int &r)
{
   return Vec8bf(svcmplt_n_s32(svptrue_b32(), l.data, r));
}

static inline Vec8bf operator<(const int l, const Vec8i &r)
{
   return Vec8bf(svcmpgt_n_s32(svptrue_b32(), r.data, l));
}
static inline Vec8bf operator<(const Vec8i &l, const Vec8i &r)
{
   return Vec8bf(svcmplt_s32(svptrue_b32(), l.data, r.data));
}

static inline Vec8bf operator<=(const Vec8i &l, const Vec8i &r)
{
   return Vec8bf(svcmple_s32(svptrue_b32(), l.data, r.data));
}

static inline Vec8bf operator<=(const int l, const Vec8i &r)
{
   return Vec8bf(svcmpge_n_s32(svptrue_b32(), r.data, l));
}

static inline Vec8i min(const Vec8i &l, const Vec8i &r)
{
   return Vec8i(svmin_s32_x(svptrue_b32(), l.data, r.data));
}

static inline Vec8i min(int const &l, Vec8i const &r)
{
   return Vec8i(svmin_n_s32_x(svptrue_b32(), r.data, l));
}

static inline Vec8i min(Vec8i const &r, int const &l)
{
   return Vec8i(svmin_n_s32_x(svptrue_b32(), r.data, l));
}

static inline Vec8i max(const Vec8i &l, const Vec8i &r)
{
   return Vec8i(svmax_s32_x(svptrue_b32(), l.data, r.data));
}

static inline Vec8i max(int const &l, Vec8i const &r)
{
   return Vec8i(svmax_n_s32_x(svptrue_b32(), r.data, l));
}

static inline Vec8i max(Vec8i const &r, int const &l)
{
   return Vec8i(svmax_n_s32_x(svptrue_b32(), r.data, l));
}

static inline Vec8i select(Vec8bf const &a, Vec8i const &b, Vec8i const &c)
{
   return Vec8i(svsel_s32(a.data, b.data, c.data));
}

static inline Vec8i select(Vec8bf const &a, int const &b, Vec8i const &c)
{
   return Vec8i(
       a[0] ? b : c[0],
       a[1] ? b : c[1],
       a[2] ? b : c[2],
       a[3] ? b : c[3],
       a[4] ? b : c[4],
       a[5] ? b : c[5],
       a[6] ? b : c[6],
       a[7] ? b : c[7]);
}

static inline Vec8i select(Vec8bf const &a, Vec8i const &b, int const &c)
{
   return Vec8i(
       a[0] ? b[0] : c,
       a[1] ? b[1] : c,
       a[2] ? b[2] : c,
       a[3] ? b[3] : c,
       a[4] ? b[4] : c,
       a[5] ? b[5] : c,
       a[6] ? b[6] : c,
       a[7] ? b[7] : c

   );
}

static inline Vec8i select(Vec8bf const &a, int const &b, int const &c)
{
   return Vec8i(
       a[0] ? b : c,
       a[1] ? b : c,
       a[2] ? b : c,
       a[3] ? b : c,
       a[4] ? b : c,
       a[5] ? b : c,
       a[6] ? b : c,
       a[7] ? b : c);
}

static inline Vec8bf operator&&(const Vec8bf &l, const Vec8bf &r)
{
   return Vec8bf(svand_b_z(svptrue_b32(), l.data, r.data));
}

static inline Vec8bf operator||(const Vec8bf &l, const Vec8bf &r)
{
   return Vec8bf(svorr_b_z(svptrue_b32(), l.data, r.data));
}

static inline bool horizontal_or(Vec8bf const &a)
{
   return svptest_any(svptrue_b32(), a.data);
}

static inline bool horizontal_and(Vec8bf const &a)
{
   return (svcntp_b32(svptrue_b32(), a.data) == 8);
}

static inline Vec8bf operator!(const Vec8bf &l)
{
   return Vec8bf(
       !l[0],
       !l[1],
       !l[2],
       !l[3],
       !l[4],
       !l[5],
       !l[6],
       !l[7]);
}
#endif


#ifdef VEC4F_SVE_SIPEARL
// #################################################################

// Vec4i implementation

class Vec4i
{
public:
   svfixed_int32x4_t data;
   // Default constructor:
   Vec4i()
   {
   }
   Vec4i(svfixed_int32x4_t x)
   {
      data = x;
   }
   // Constructor to broadcast the same value into all elements:
   Vec4i(int f)
   {
      data = svdup_n_s32(f);
   }
   // Replicate 4 values across v.
   Vec4i(int a, int b, int c, int d)
   {
      int val[4] = {a, b, c, d};
      data = svld1(svptrue_b32(), val);
   }
   Vec4i(Vec4i const &x)
   {
      data = x.data;
   }
   // Member function to load from array (unaligned)
   Vec4i &load(int const *p)
   {
      data = svld1(svptrue_b32(), p);
      return *this;
   }

   // Member function to load from array, aligned by 32
   Vec4i &load_a(int const *p)
   {
      return this->load(p);
   }

   Vec4i &insert(int i, int const &x)
   {
      int aux[4];
      svst1_s32(svptrue_b32(), aux, data);
      aux[i] = x;
      data = svld1(svptrue_b32(), aux);
      return *this;
   }

   // Member function to store into array (unaligned)
   void store(int *x) const
   {
      svst1_s32(svptrue_b32(), x, data);
   }

   // Member function to store into array, aligned by 32
   void store_a(int *p) const
   {
      this->store(p);
   }

   Vec4i &operator=(Vec4i const &r)
   {
      data = r.data;
      return *this;
   }
   int extract(int index) const
   {
      return svclastb_n_s32(svwhilele_b32_s32(0, index), 0, data);
   }

   int operator[](int index) const
   {
      return svclastb_n_s32(svwhilele_b32_s32(0, index), 0, data);
   }

   Vec4i operator++(int)
   {
      data = svadd_n_s32_x(svptrue_b32(), this->data, 1);
      return *this;
   }
};

class Vec4f
{
public:
   svfixed_float32x4_t data;
   // Default constructor:
   Vec4f()
   {
   }
   Vec4f(svfixed_float32x4_t x)
   {
      data = x;
   }
   // Constructor to broadcast the same value into all elements:
   Vec4f(float f)
   {
      data = svdup_n_f32(f);
   }
   // Replicate 8 values across v.
   Vec4f(float a, float b, float c, float d)
   {
      float val[4] = {a, b, c, d};
      data = svld1(svptrue_b32(), &val[0]);
   }
   // Copy vector v.
   Vec4f(Vec4f const &x)
   {
      data = x.data;
   }
   // Member function to load from array (unaligned)
   Vec4f &load(float const *p)
   {
      data = svld1(svptrue_b32(), p);
      return *this;
   }

   // Member function to load from array, aligned by 32
   Vec4f &load_a(float const *p)
   {
      return this->load(p);
   }

   Vec4f &insert(int i, float const &x)
   {
      float aux[4];
      svst1_f32(svptrue_b32(), aux, data);
      aux[i] = x;
      data = svld1(svptrue_b32(), aux);
      return *this;
   }

   // Member function to store into array (unaligned)
   void store(float *x) const
   {
      svst1_f32(svptrue_b32(), x, data);
   }

   // Member function to store into array, aligned by 32
   void store_a(float *p) const
   {
      this->store(p);
   }

   Vec4f &operator=(Vec4f const &r)
   {
      data = r.data;
      return *this;
   }

   float extract(int index) const
   {
      return svclastb_n_f32(svwhilele_b32_u32(0, index), 0.0f, data);
   }

   float operator[](int index) const
   {
      return svclastb_n_f32(svwhilele_b32_u32(0, index), 0.0f, data);
   }

   Vec4f operator++(int)
   {
      data = svadd_n_f32_x(svptrue_b32(), this->data, 1.0f);
      return *this;
   }
};

// #################################################################

// #################################################################

// Vec4bf implementation
class Vec4bf
{
public:
   svfixed_predicatex4_t data;
   // Default constructor:
   Vec4bf()
   {
   }
   Vec4bf(svfixed_predicatex4_t x)
   {
      data = x;
   }

   Vec4bf(bool x0, bool x1, bool x2, bool x3)
   {
      Vec4f veca = {static_cast<float>(x0), static_cast<float>(x1), static_cast<float>(x2), static_cast<float>(x3)};
      data = svcmpne_f32(svptrue_b32(), veca.data, svdup_n_f32(0.0f));
   }

   bool operator[](int index) const
   {
      Vec4f result = svsel_f32(data, svdup_n_f32(1.0f), svdup_n_f32(0.0f));
      return result[index] == 0.0f ? false : true;
   }
};

static inline Vec4f abs(const Vec4f &l)
{
   return Vec4f(svabs_f32_x(svptrue_b32(), l.data));
}

static inline Vec4f sqrt(const Vec4f &l)
{
   return Vec4f(svsqrt_f32_x(svptrue_b32(), l.data));
}

static inline Vec4f operator+(Vec4f const &a, Vec4f const &b)
{
   return Vec4f(svadd_f32_x(svptrue_b32(), a.data, b.data));
}

static inline Vec4f operator+(Vec4f const &a, float b)
{
   return Vec4f(svadd_n_f32_x(svptrue_b32(), a.data, b));
}

static inline Vec4f operator+(float a, Vec4f const &b)
{
   return Vec4f(svadd_n_f32_x(svptrue_b32(), b.data, a));
}

static inline Vec4f operator-(const Vec4f &r)
{
   return Vec4f(svneg_f32_x(svptrue_b32(), r.data));
}

static inline Vec4f operator-(Vec4f const &a, Vec4f const &b)
{
   return Vec4f(svsub_f32_x(svptrue_b32(), a.data, b.data));
}

static inline Vec4f operator-(Vec4f const &a, float b)
{
   return Vec4f(svsub_n_f32_x(svptrue_b32(), a.data, b));
}

static inline Vec4f operator-(float a, Vec4f const &b)
{
   return Vec4f(svsub_f32_x(svptrue_b32(), svdup_n_f32(a), b.data));
}

static inline Vec4f operator*(Vec4f const &a, Vec4f const &b)
{
   return Vec4f(svmul_f32_x(svptrue_b32(), a.data, b.data));
}

static inline Vec4f operator*(Vec4f const &a, float b)
{
   return Vec4f(svmul_n_f32_x(svptrue_b32(), a.data, b));
}

static inline Vec4f operator*(float a, Vec4f const &b)
{
   return Vec4f(svmul_n_f32_x(svptrue_b32(), b.data, a));
}

static inline Vec4f operator/(const Vec4f &l, const Vec4f &r)
{
   return Vec4f(svdiv_f32_x(svptrue_b32(), l.data, r.data));
}

static inline Vec4f operator/(const Vec4f &l, const float &r)
{
   return Vec4f(svdiv_n_f32_x(svptrue_b32(), l.data, r));
}

static inline Vec4f operator/(const float &r, const Vec4f &l)
{
   return Vec4f(svdiv_f32_x(svptrue_b32(), svdup_n_f32(r), l.data));
}

static inline Vec4f &operator+=(Vec4f &l, const Vec4f &r)
{
   l.data = svadd_f32_x(svptrue_b32(), l.data, r.data);
   return l;
}

static inline Vec4f &operator+=(Vec4f &l, const float &r)
{
   l.data = svadd_f32_x(svptrue_b32(), l.data, svdup_n_f32(r));
   return l;
}

static inline Vec4f &operator-=(Vec4f &l, const Vec4f &r)
{
   l = l - r;
   return l;
}

static inline Vec4f &operator-=(Vec4f &l, const float &r)
{
   l = l - r;
   return l;
}

// operator ||

// operator &&

static inline Vec4bf operator==(const Vec4f &l, const Vec4f &r)
{
   return Vec4bf(svcmpeq_f32(svptrue_b32(), l.data, r.data));
}

static inline Vec4bf operator!=(const Vec4f &l, const Vec4f &r)
{
   return Vec4bf(svcmpne_f32(svptrue_b32(), l.data, r.data));
}

static inline Vec4bf operator!=(const Vec4f &l, const float &r)
{
   return Vec4bf(svcmpne_n_f32(svptrue_b32(), l.data, r));
}

static inline Vec4bf operator>(const Vec4f &l, const Vec4f &r)
{
   return Vec4bf(svcmpgt_f32(svptrue_b32(), l.data, r.data));
}

static inline Vec4bf operator>(const Vec4f &l, const float r)
{
   return Vec4bf(svcmpgt_n_f32(svptrue_b32(), l.data, r));
}

static inline Vec4bf operator>(const float l, const Vec4f &r)
{
   return Vec4bf(svcmplt_n_f32(svptrue_b32(), r.data, l));
}

static inline Vec4bf operator>=(const Vec4f &l, const Vec4f &r)
{
   return Vec4bf(svcmpge_f32(svptrue_b32(), l.data, r.data));
}

static inline Vec4bf operator>=(const Vec4f &l, const float r)
{
   return Vec4bf(svcmple_n_f32(svptrue_b32(), l.data, r));
}

static inline Vec4bf operator>=(const float l, const Vec4f &r)
{
   return Vec4bf(svcmpge_n_f32(svptrue_b32(), r.data, l));
}

static inline Vec4bf operator<(const Vec4f &l, const float &r)
{
   return Vec4bf(svcmplt_n_f32(svptrue_b32(), l.data, r));
}

static inline Vec4bf operator<(const float l, const Vec4f &r)
{
   return Vec4bf(svcmpgt_n_f32(svptrue_b32(), r.data, l));
}
static inline Vec4bf operator<(const Vec4f &l, const Vec4f &r)
{
   return Vec4bf(svcmplt_f32(svptrue_b32(), l.data, r.data));
}

static inline Vec4bf operator<=(const Vec4f &l, const Vec4f &r)
{
   return Vec4bf(svcmple_f32(svptrue_b32(), l.data, r.data));
}

static inline Vec4bf operator<=(const float l, const Vec4f &r)
{
   return Vec4bf(svcmpge_n_f32(svptrue_b32(), r.data, l));
}

static inline Vec4f min(const Vec4f &l, const Vec4f &r)
{
   return Vec4f(svmin_f32_x(svptrue_b32(), l.data, r.data));
}

static inline Vec4f min(float const &l, Vec4f const &r)
{
   return Vec4f(svmin_n_f32_x(svptrue_b32(), r.data, l));
}

static inline Vec4f min(Vec4f const &r, float const &l)
{
   return Vec4f(svmin_n_f32_x(svptrue_b32(), r.data, l));
}

static inline Vec4f max(const Vec4f &l, const Vec4f &r)
{
   return Vec4f(svmax_f32_x(svptrue_b32(), l.data, r.data));
}

static inline Vec4f max(float const &l, Vec4f const &r)
{
   return Vec4f(svmax_n_f32_x(svptrue_b32(), r.data, l));
}

static inline Vec4f max(Vec4f const &r, float const &l)
{
   return Vec4f(svmax_n_f32_x(svptrue_b32(), r.data, l));
}

static inline Vec4f select(Vec4bf const &a, Vec4f const &b, Vec4f const &c)
{
   return Vec4f(svsel_f32(a.data, b.data, c.data));
}

static inline Vec4f select(Vec4bf const &a, float const &b, Vec4f const &c)
{

   return Vec4f(svsel_f32(a.data, svdup_n_f32(b), c.data));
}

static inline Vec4f select(Vec4bf const &a, Vec4f const &b, float const &c)
{
   return Vec4f(svsel_f32(a.data, b.data, svdup_n_f32(c)));
}

static inline Vec4f select(Vec4bf const &a, float const &b, float const &c)
{
   return Vec4f(svsel_f32(a.data, svdup_n_f32(b), svdup_n_f32(c)));
}

static inline Vec4i abs(const Vec4i &l)
{
   return Vec4i(svabs_s32_x(svptrue_b32(), l.data));
}

static inline Vec4i operator+(Vec4i const &a, Vec4i const &b)
{
   return Vec4i(svadd_s32_x(svptrue_b32(), a.data, b.data));
}

// vector operator + : add vector and scalar
static inline Vec4i operator+(Vec4i const &a, int b)
{
   return Vec4i(svadd_n_s32_x(svptrue_b32(), a.data, b));
}

static inline Vec4i operator+(int a, Vec4i const &b)
{
   return Vec4i(svadd_n_s32_x(svptrue_b32(), b.data, a));
}

static inline Vec4i operator-(const Vec4i &r)
{
   return Vec4i(svneg_s32_x(svptrue_b32(), r.data));
}

static inline Vec4i operator-(Vec4i const &a, Vec4i const &b)
{
   return Vec4i(svsub_s32_x(svptrue_b32(), a.data, b.data));
}

static inline Vec4i operator-(Vec4i const &a, int b)
{
   return Vec4i(svsub_n_s32_x(svptrue_b32(), a.data, b));
}

static inline Vec4i operator-(int a, Vec4i const &b)
{
   return Vec4i(svsub_s32_x(svptrue_b32(), (svptrue_b32(), svdup_n_s32(a)), b.data));
}

static inline Vec4i operator*(Vec4i const &a, Vec4i const &b)
{
   return Vec4i(svmul_s32_x(svptrue_b32(), a.data, b.data));
}

static inline Vec4i operator*(Vec4i const &a, int b)
{
   return Vec4i(svmul_n_s32_x(svptrue_b32(), a.data, b));
}

static inline Vec4i operator*(int a, Vec4i const &b)
{
   return Vec4i(svmul_n_s32_x(svptrue_b32(), b.data, a));
}

static inline Vec4i operator/(const Vec4i &l, const Vec4i &r)
{
   return Vec4i(svdiv_s32_x(svptrue_b32(), l.data, r.data));
}

static inline Vec4i operator/(const Vec4i &l, const int &r)
{
   return Vec4i(svdiv_n_s32_x(svptrue_b32(), l.data, r));
}

static inline Vec4i operator/(const int &r, const Vec4i &l)
{
   return Vec4i(svdiv_s32_x(svptrue_b32(), (svptrue_b32(), svdup_n_s32(r)), l.data));
}

static inline Vec4i &operator+=(Vec4i &l, const Vec4i &r)
{
   l = l + r;
   return l;
}

static inline Vec4i &operator+=(Vec4i &l, const int &r)
{
   l = l + r;
   return l;
}

static inline Vec4i &operator-=(Vec4i &l, const Vec4i &r)
{
   l = l - r;
   return l;
}

static inline Vec4i &operator-=(Vec4i &l, const int &r)
{
   l = l - r;
   return l;
}

// operator ||

// operator &&

static inline Vec4bf operator==(const Vec4i &l, const Vec4i &r)
{
   return Vec4bf(svcmpeq_s32(svptrue_b32(), l.data, r.data));
}

static inline Vec4bf operator!=(const Vec4i &l, const Vec4i &r)
{
   return Vec4bf(svcmpne_s32(svptrue_b32(), l.data, r.data));
}

static inline Vec4bf operator!=(const Vec4i &l, const int &r)
{
   return Vec4bf(svcmpne_n_s32(svptrue_b32(), l.data, r));
}

static inline Vec4bf operator>(const Vec4i &l, const Vec4i &r)
{
   return Vec4bf(svcmpgt_s32(svptrue_b32(), l.data, r.data));
}

static inline Vec4bf operator>(const Vec4i &l, const int r)
{
   return Vec4bf(svcmpgt_n_s32(svptrue_b32(), l.data, r));
}

static inline Vec4bf operator>(const int l, const Vec4i &r)
{
   return Vec4bf(svcmplt_n_s32(svptrue_b32(), r.data, l));
}

static inline Vec4bf operator>=(const Vec4i &l, const Vec4i &r)
{
   return Vec4bf(svcmpge_s32(svptrue_b32(), l.data, r.data));
}

static inline Vec4bf operator>=(const Vec4i &l, const int r)
{
   return Vec4bf(svcmple_n_s32(svptrue_b32(), l.data, r));
}

static inline Vec4bf operator>=(const int l, const Vec4i &r)
{
   return Vec4bf(svcmpge_n_s32(svptrue_b32(), r.data, l));
}

static inline Vec4bf operator<(const Vec4i &l, const int &r)
{
   return Vec4bf(svcmplt_n_s32(svptrue_b32(), l.data, r));
}

static inline Vec4bf operator<(const int l, const Vec4i &r)
{
   return Vec4bf(svcmpgt_n_s32(svptrue_b32(), r.data, l));
}
static inline Vec4bf operator<(const Vec4i &l, const Vec4i &r)
{
   return Vec4bf(svcmplt_s32(svptrue_b32(), l.data, r.data));
}

static inline Vec4bf operator<=(const Vec4i &l, const Vec4i &r)
{
   return Vec4bf(svcmple_s32(svptrue_b32(), l.data, r.data));
}

static inline Vec4bf operator<=(const int l, const Vec4i &r)
{
   return Vec4bf(svcmpge_n_s32(svptrue_b32(), r.data, l));
}

static inline Vec4i min(const Vec4i &l, const Vec4i &r)
{
   return Vec4i(svmin_s32_x(svptrue_b32(), l.data, r.data));
}

static inline Vec4i min(int const &l, Vec4i const &r)
{
   return Vec4i(svmin_n_s32_x(svptrue_b32(), r.data, l));
}

static inline Vec4i min(Vec4i const &r, int const &l)
{
   return Vec4i(svmin_n_s32_x(svptrue_b32(), r.data, l));
}

static inline Vec4i max(const Vec4i &l, const Vec4i &r)
{
   return Vec4i(svmax_s32_x(svptrue_b32(), l.data, r.data));
}

static inline Vec4i max(int const &l, Vec4i const &r)
{
   return Vec4i(svmax_n_s32_x(svptrue_b32(), r.data, l));
}

static inline Vec4i max(Vec4i const &r, int const &l)
{
   return Vec4i(svmax_n_s32_x(svptrue_b32(), r.data, l));
}

static inline Vec4i select(Vec4bf const &a, Vec4i const &b, Vec4i const &c)
{
   return Vec4i(svsel_s32(a.data, b.data, c.data));
}

static inline Vec4i select(Vec4bf const &a, int const &b, Vec4i const &c)
{
   return Vec4i(
       a[0] ? b : c[0],
       a[1] ? b : c[1],
       a[2] ? b : c[2],
       a[3] ? b : c[3]
   );
}

static inline Vec4i select(Vec4bf const &a, Vec4i const &b, int const &c)
{
   return Vec4i(
       a[0] ? b[0] : c,
       a[1] ? b[1] : c,
       a[2] ? b[2] : c,
       a[3] ? b[3] : c
   );
}

static inline Vec4i select(Vec4bf const &a, int const &b, int const &c)
{
   return Vec4i(
       a[0] ? b : c,
       a[1] ? b : c,
       a[2] ? b : c,
       a[3] ? b : c
   );
}

static inline Vec4bf operator&&(const Vec4bf &l, const Vec4bf &r)
{
   return Vec4bf(svand_b_z(svptrue_b32(), l.data, r.data));
}

static inline Vec4bf operator||(const Vec4bf &l, const Vec4bf &r)
{
   return Vec4bf(svorr_b_z(svptrue_b32(), l.data, r.data));
}

static inline bool horizontal_or(Vec4bf const &a)
{
   return svptest_any(svptrue_b32(), a.data);
}

static inline bool horizontal_and(Vec4bf const &a)
{
   return (svcntp_b32(svptrue_b32(), a.data) == 8);
}

static inline Vec4bf operator!(const Vec4bf &l)
{
   return Vec4bf(
       !l[0],
       !l[1],
       !l[2],
       !l[3]
       );
}

// #################################################################

static inline Vec4i truncate_to_int(Vec4f const &a)
{
   return Vec4i(a[0], a[1], a[2], a[3]);
}

// TODO Vec4D implementation
// static inline Vec4i to_double(Vec4f const & a){
//   return Vec4d(a[0], a[1], a[2], a[3],
//                          a[4], a[5], a[6], a[7]);
//}

static inline Vec4f to_float(Vec4i const &a)
{
   return Vec4f(a[0], a[1], a[2], a[3]);
}
#endif



#ifdef VEC8F_SVE_SIPEARL
// #################################################################

static inline Vec8i truncate_to_int(Vec8f const &a)
{
   return Vec8i(a[0], a[1], a[2], a[3],
                a[4], a[5], a[6], a[7]);
}

// TODO Vec8D implementation
// static inline Vec8Simple<double> to_double(Vec8f const & a){
//   return Vec8Simple<double>(a[0], a[1], a[2], a[3],
//                          a[4], a[5], a[6], a[7]);
//}

static inline Vec8f to_float(Vec8i const &a)
{
   return Vec8f(a[0], a[1], a[2], a[3],
                a[4], a[5], a[6], a[7]);
}
#endif

#endif