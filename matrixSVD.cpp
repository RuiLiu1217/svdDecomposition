
#include <cstdlib>
#include <cmath>
#include <math.h>
#include <utility>
#include <vector_types.h>
#include <vector_functions.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <functional>
#include <algorithm>
#include <omp.h>

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <utility>
#include "mkl.h"
#include <assert.h>

namespace genSysMatrix
{
  int calculateSVD(int rowNum, int colNum, float* A, float* U, float* Sigma, float* Vt) {
    int lda = rowNum;
    int ldu = rowNum;
    int ldvt = colNum;
    float* superb = new float[std::min(rowNum, colNum)-1];
    int info = LAPACKE_sgesvd( LAPACK_COL_MAJOR, 'A', 'A', rowNum, colNum, A, lda,
			       Sigma, U, ldu, Vt, ldvt, superb);
    if(info > 0) {
      std::cout<<"The algorithm computing SVD failed to converge\n";
    }
    delete[] superb;
    return info;
  }
  static const float PI = 3.14159265359f;
  static const float EPSILON = 1.0E-9;
	
  /// \brief Fan Beam Equal Angle Detector based CT system
  class FanEAGeo
  {
  public:
    float m_S2O; ///< source to object distance
    float m_O2D; ///< object to detector distance
    float m_S2D; ///< source to detector distance
    int m_ViwN; ///< view number
    float m_ViwBeg; ///< Begin viewer number
    float m_ViwStp; ///< View step size

    float m_DetArc; ///< Detector Arc angle
    int m_DetN; ///< Detector cells number in one row
    float m_DetStp; ///< Detector cells size
    float m_DetCntIdx; ///< The index of the center of the detector
  public:
    FanEAGeo(void);
    ~FanEAGeo(void) {};
    FanEAGeo(const FanEAGeo& rhs);
    /// \brief constructor
    /// \param S2O source to object distance
    /// \param O2D object to detector distance
    /// \param ViwN view number
    /// \param ViwBeg the begin view
    /// \param ViwEnd the End view
    /// \param DetArc the detector Arc
    /// \param DetN the number of detector cells on one row
    FanEAGeo(const float S2O, const float O2D, const  int ViwN,
	     const float ViwBeg, const float ViwEnd, const float DetArc,
	     const  int DetN);
  };


  FanEAGeo::FanEAGeo(void)
  {
    m_DetArc = 0.95928517242269f / 2;
    m_DetN = 444;
    m_DetCntIdx = 222;
    m_DetStp = m_DetArc / m_DetN;
    m_O2D = 4.082259521484375e+02;

    m_S2O = 5.385200195312500e+02;
    m_S2D = m_S2O + m_O2D;
    m_ViwBeg = 0.0f;// (-2.668082275390625e+02 / 180.0* 3.14159265358979323846264);
    m_ViwN = 61;
    m_ViwStp = 3.14159265358979323846264f * 2.0f / m_ViwN;
  }

  FanEAGeo::FanEAGeo(const FanEAGeo& rhs)
  {
    m_DetArc = rhs.m_DetArc;
    m_DetN = rhs.m_DetN;
    m_DetCntIdx = rhs.m_DetCntIdx;
    m_DetStp = rhs.m_DetStp;
    m_O2D = rhs.m_O2D;
    m_S2O = rhs.m_S2O;
    m_S2D = rhs.m_S2D;
    m_ViwBeg = rhs.m_ViwBeg;
    m_ViwN = rhs.m_ViwN;
    m_ViwStp = rhs.m_ViwStp;
  }


  FanEAGeo::FanEAGeo(const float S2O, const float O2D, const  int ViwN,
		     const float ViwBeg, const float ViwEnd, const float DetArc,
		     const  int DetN) :m_S2O(S2O), m_O2D(O2D), m_S2D(S2O + O2D),
				       m_ViwN(ViwN), m_ViwBeg(ViwBeg), m_ViwStp((ViwEnd - ViwBeg) / float(ViwN)),
				       m_DetArc(DetArc), m_DetN(DetN), m_DetStp(DetArc / DetN),
				       m_DetCntIdx(DetN * 0.5f - 0.5f) {}


  class FanEDGeo
  {
  public:
    float m_S2O; ///< Source to Object distance
    float m_O2D; ///< Object to Detector distance
    float m_S2D; ///< Source to Detector distance

    int m_ViwN; ///< view number
    float m_ViwBeg;///< View begin
    float m_ViwStp; ///< View Step

    float m_DetSize; ///< Detector size;
    int m_DetN; ///< How many detector cells
    float m_DetStp; ///< detector cells size;
    float m_DetCntIdx; ///< detector center index;
  public:
    /// \brief constructor
    FanEDGeo(void);
    /// \brief destructor
    ~FanEDGeo(){};
    /// \brief copy constructor
    FanEDGeo(const FanEDGeo& rhs);
    /// \brief Constructor
    /// \param S2O source to object distance
    /// \param O2D object to detector distance
    /// \param ViwN View number
    /// \param ViwBeg The begin of the view
    /// \param ViwEnd The end of the view
    /// \param DetSize Detector size
    /// \param DetN detector cells number on one row
    FanEDGeo(const float S2O, const float O2D, const  int ViwN,
	     const float ViwBeg, const float ViwEnd, const float DetSize,
	     const  int DetN);

  };


  FanEDGeo::FanEDGeo(void)
  {
    m_S2O = 40.0f;
    m_O2D = 40.0f;
    m_S2D = 80.0f;

    m_ViwN = 720; // view number
    m_ViwBeg = 0.0f;
    m_ViwStp = 3.14159265358979323846264f * 2.0f / m_ViwN;

    m_DetSize = 24.0f; // Detector size;
    m_DetN = 888; // How many detector cells
    m_DetStp = m_DetSize / m_DetN; // detector cells size;
    m_DetCntIdx = m_DetN * 0.5f - 0.5f; //detector center index;
  }

  FanEDGeo::FanEDGeo(const FanEDGeo& rhs)
  {
    m_S2O = rhs.m_S2O;
    m_O2D = rhs.m_O2D;
    m_S2D = rhs.m_S2D;

    m_ViwN = rhs.m_ViwN;
    m_ViwBeg = rhs.m_ViwBeg;
    m_ViwStp = rhs.m_ViwStp;

    m_DetSize = rhs.m_DetSize;
    m_DetN = rhs.m_DetN;
    m_DetStp = rhs.m_DetStp;
    m_DetCntIdx = rhs.m_DetCntIdx;
  }


  FanEDGeo::FanEDGeo(const float S2O, const float O2D, const  int ViwN,
		     const float ViwBeg, const float ViwEnd, const float DetSize,
		     const  int DetN) :m_S2O(S2O), m_O2D(O2D), m_S2D(S2O + O2D),
				       m_ViwN(ViwN), m_ViwBeg(ViwBeg), m_ViwStp((ViwEnd - ViwBeg) / ViwN),
				       m_DetSize(DetSize), m_DetN(DetN), m_DetStp(DetSize / DetN),
				       m_DetCntIdx(DetN * 0.5f - 0.5f){}



  /// \brief Image configuration class
  class Image
  {
  public:
    int2 m_Reso; ///< Image resolution
    float2 m_Size;///< Image size
    float2 m_Step; ///< Image Step
    float2 m_Bias; ///< The bias of the image
  public:
    /// \brief constructor
    Image(void);
    /// \brief destructor
    ~Image(void) {};
    /// \brief copy constructor
    Image(const Image& rhs);
    /// \brief constructor
    Image(
	  const int resoL,///< resolution on length direction
	  const int resoW,///< resolution on width direction
	  const float sizeL, ///< length size of the image
	  const float sizeW,///< width size of the image
	  const float BiasL, ///< bias on length direction
	  const float BiasW ///<bias on width direction
	  );
  };
	
  Image::Image(void)
  {
    m_Bias.x = 0.0f;
    m_Bias.y = 0.0f;  //ÕâžöÆ«ÒÆµÄµ¥Î»ÊÇÕæÊµÎïÀíµ¥Î»;
    m_Reso.x = m_Reso.y = 512;
    m_Size.x = m_Size.y = 4.484740011196460e+02;
    m_Step.x = m_Size.x / m_Reso.x;
    m_Step.y = m_Size.y / m_Reso.y;
  }

  Image::Image(const Image& rhs)
  {
    m_Bias = rhs.m_Bias;
    m_Reso = rhs.m_Reso;
    m_Size = rhs.m_Size;
    m_Step = rhs.m_Step;
  }



  Image::Image(
	       const int resoL,
	       const int resoW,
	       const float sizeL,
	       const float sizeW,
	       const float BiasL,
	       const float BiasW) :m_Reso(make_int2(resoL, resoW)),
				   m_Size(make_float2(sizeL, sizeW)),
				   m_Step(make_float2(sizeL / resoL, sizeW / resoW)),
				   m_Bias(make_float2(BiasL, BiasW)) {}


  template<typename T>
  inline bool IS_ZERO(const T& x)
  {
    return ((x < EPSILON) && (x > -EPSILON));
  }



  float2 rotation(const float2& p, const float& cosT, const float& sinT)
  {
    float2 curP;
    curP.x = p.x * cosT - p.y * sinT;
    curP.y = p.x * sinT + p.y * cosT;
    return curP;
  }

  double2 rotation(const double2& p, const double& cosT, const double& sinT)
  {
    double2 curP;
    curP.x = p.x * cosT - p.y * sinT;
    curP.y = p.x * sinT + p.y * cosT;
    return curP;
  }

  struct Ray2D
  {
  public:
    float2 o;
    float2 d;
  };

  struct Ray2Dd
  {
  public:
    double2 o;
    double2 d;
  };


  double2 operator-(const double2& a, const double2& b)
  {
    double2 res;
    res.x = a.x - b.x;
    res.y = a.y - b.y;
    return res;
  }

  float2 operator-(const float2& a, const float2& b)
  {
    float2 res;
    res.x = a.x - b.x;
    res.y = a.y - b.y;
    return res;
  }

  double2 normalize(const double2& o)
  {
    double l = hypot(o.x, o.y);
    double2 res;
    res.x = o.x / l;
    res.y = o.y / l;
    return res;
  }


  float2 normalize(const float2& o)
  {
    float l = hypot(o.x, o.y);
    float2 res;
    res.x = o.x / l;
    res.y = o.y / l;
    return res;
  }

  inline float dev_pFun(const float& alpha, const float& pstart, const float&pend)
  {
    return pstart + alpha * (pend - pstart);
  }


  inline double dev_pFun(const double& alpha, const double& pstart, const double&pend)
  {
    return pstart + alpha * (pend - pstart);
  }


  inline float dev_alpha_IFun(const float& b, const float& d, const float& pstart, const float& pend, const unsigned int& i)
  {
    if (!IS_ZERO(pend - pstart))
      {
	return ((b + (float) i*d) - pstart) / (pend - pstart);
      }
    else return 1000;//((b + i*d)-pstart)/(1e-6);
  }

  inline double dev_alpha_IFun(const double& b, const double& d, const double& pstart, const double& pend, const unsigned int& i)
  {
    if (!IS_ZERO(pend - pstart))
      {
	return ((b + (double) i*d) - pstart) / (pend - pstart);
      }
    else return 1000;//((b + i*d)-pstart)/(1e-6);
  }


  inline float dev_varphiFun(const float& alpha, const float& b, const float& d, const float& pstart, const float& pend)
  {
    return (dev_pFun(alpha, pstart, pend) - b) / d;
  }


  inline double dev_varphiFun(const double& alpha, const double& b, const double& d, const double& pstart, const double& pend)
  {
    return (dev_pFun(alpha, pstart, pend) - b) / d;
  }


  inline void dev_minmaxIdxFun(
			       const float& pstart, const float& pend,
			       const float& b, const float& d,
			       const float& alphaMIN, const float& alphaMAX,
			       const float& alphaPmin, const float& alphaPmax,
			       const unsigned int& Nplane, int* imin, int* imax)
  {
    if (pstart < pend)
      {
	if (IS_ZERO(alphaMIN - alphaPmin))
	  {
	    *imin = 1;
	  }
	else
	  {
	    *imin = static_cast<int>(ceil(dev_varphiFun(alphaMIN, b, d, pstart, pend)));
	  }
	if (IS_ZERO(alphaMAX - alphaPmax))
	  {
	    *imax = Nplane - 1;
	  }
	else
	  {
	    *imax = static_cast<int>(dev_varphiFun(alphaMAX, b, d, pstart, pend));
	  }
      }
    else
      {
	if (IS_ZERO(alphaMIN - alphaPmin))
	  {
	    *imax = Nplane - 2;
	  }
	else
	  {
	    *imax = static_cast<int>(dev_varphiFun(alphaMIN, b, d, pstart, pend));
	  }
	if (IS_ZERO(alphaMAX - alphaPmax))
	  {
	    *imin = 0;
	  }
	else
	  {
	    *imin = static_cast<int>(ceil(dev_varphiFun(alphaMAX, b, d, pstart, pend)));
	  }
      }
  }


  inline void dev_minmaxIdxFun(
			       const double& pstart, const double& pend,
			       const double& b, const double& d,
			       const double& alphaMIN, const double& alphaMAX,
			       const double& alphaPmin, const double& alphaPmax,
			       const unsigned int& Nplane, int* imin, int* imax)
  {
    if (pstart < pend)
      {
	if (IS_ZERO(alphaMIN - alphaPmin))
	  {
	    *imin = 1;
	  }
	else
	  {
	    *imin = static_cast<int>(ceil(dev_varphiFun(alphaMIN, b, d, pstart, pend)));
	  }
	if (IS_ZERO(alphaMAX - alphaPmax))
	  {
	    *imax = Nplane - 1;
	  }
	else
	  {
	    *imax = static_cast<int>(dev_varphiFun(alphaMAX, b, d, pstart, pend));
	  }
      }
    else
      {
	if (IS_ZERO(alphaMIN - alphaPmin))
	  {
	    *imax = Nplane - 2;
	  }
	else
	  {
	    *imax = static_cast<int>(dev_varphiFun(alphaMIN, b, d, pstart, pend));
	  }
	if (IS_ZERO(alphaMAX - alphaPmax))
	  {
	    *imin = 0;
	  }
	else
	  {
	    *imin = static_cast<int>(ceil(dev_varphiFun(alphaMAX, b, d, pstart, pend)));
	  }
      }
  }



  inline float dev_alphaU_Fun(const float& d, const float& startx, const float& endx)
  {
    if (IS_ZERO(startx - endx))
      {
	return 1000.0f;//(d/1e-6);
      }
    return d / abs(startx - endx);
  }

  inline double dev_alphaU_Fun(const double& d, const double& startx, const double& endx)
  {
    if (IS_ZERO(startx - endx))
      {
	return 1000.0f;//(d/1e-6);
      }
    return d / abs(startx - endx);
  }


  inline  int dev_iu_Fun(const float& start, const float& end)
  {
    return (start < end) ? 1 : -1;
  }


  inline  int dev_iu_Fun(const double& start, const double& end)
  {
    return (start < end) ? 1 : -1;
  }



  void pushMatrix_Direct(
			 float* A,
			 float startX, float startY,
			 float endX, float endY,
			 float bx, float by,

			 float dx, float dy,
			 const int objResoLen,
			 const int objResoWid,
			 int& rowidx, int colNum, int sampleNum)
  {
    const float dirX(endX - startX);
    const float dirY(endY - startY);
    const float lengthSq = dirX * dirX + dirY * dirY;
    const float dconv = sqrt(lengthSq);
    int imin, imax, jmin, jmax;
    const float alphaxmin = fminf(dev_alpha_IFun(bx, dx, startX, endX, 0), dev_alpha_IFun(bx, dx, startX, endX, objResoLen));
    const float alphaxmax = fmaxf(dev_alpha_IFun(bx, dx, startX, endX, 0), dev_alpha_IFun(bx, dx, startX, endX, objResoLen));
    const float alphaymin = fminf(dev_alpha_IFun(by, dy, startY, endY, 0), dev_alpha_IFun(by, dy, startY, endY, objResoWid));
    const float alphaymax = fmaxf(dev_alpha_IFun(by, dy, startY, endY, 0), dev_alpha_IFun(by, dy, startY, endY, objResoWid));

    const float alphaMIN = fmaxf(alphaxmin, alphaymin);
    const float alphaMAX = fminf(alphaxmax, alphaymax);
    dev_minmaxIdxFun(startX, endX, bx, dx, alphaMIN, alphaMAX, alphaxmin, alphaxmax, objResoLen + 1, &imin, &imax);
    dev_minmaxIdxFun(startY, endY, by, dy, alphaMIN, alphaMAX, alphaymin, alphaymax, objResoWid + 1, &jmin, &jmax);

    float alphaX = (startX < endX) ? dev_alpha_IFun(bx, dx, startX, endX, imin) : dev_alpha_IFun(bx, dx, startX, endX, imax);
    float alphaY = (startY < endY) ? dev_alpha_IFun(by, dy, startY, endY, jmin) : dev_alpha_IFun(by, dy, startY, endY, jmax);

    int Np = static_cast<int>(fabsf(imax - imin + 1.0f) + fabsf(jmax - jmin + 1.0f) + 4.0f);
    const float alphaxu = dev_alphaU_Fun(dx, startX, endX);
    const float alphayu = dev_alphaU_Fun(dy, startY, endY);

    float alphaC = alphaMIN;

    int i = static_cast<int>(dev_varphiFun(alphaMIN* 1.00003f, bx, dx, startX, endX));
    int j = static_cast<int>(dev_varphiFun(alphaMIN* 1.00003f, by, dy, startY, endY));

    const int iuu = dev_iu_Fun(startX, endX);
    const int juu = dev_iu_Fun(startY, endY);

    float d12(0.0f);
    float weight(0.0f);
    unsigned int repIdx(0);
    unsigned int colidx(0);
    while (repIdx != Np)
      {
	if (i < 0 || i >= objResoLen || j < 0 || j >= objResoWid)
	  {
	    break;
	  }
	if (alphaX <= alphaY)
	  {
	    colidx = j * objResoLen + i;
	    weight = (alphaX - alphaC) * dconv;
	    A[rowidx * colNum + colidx] += weight / static_cast<float>(sampleNum);

	    // wgt.push_back(weight);
	    // rowIdx.push_back(rowidx);
	    // colIdx.push_back(colidx);

	    d12 += weight;
	    i += iuu;
	    alphaC = alphaX;
	    alphaX += alphaxu;
	  }
	else
	  {
	    colidx = j * objResoLen + i;
	    weight = (alphaY - alphaC) * dconv;
	    A[rowidx * colNum + colidx] += weight / static_cast<float>(sampleNum);

	    // wgt.push_back(weight);
	    // rowIdx.push_back(rowidx);
	    // colIdx.push_back(colidx);

	    d12 += weight;
	    j += juu;

	    alphaC = alphaY;
	    alphaY += alphayu;
	  }
	++repIdx;
      }
  }



  void genProj_SIDDON_DirectMatrix(
				   float* A,
				   const FanEAGeo& FanGeo,
				   const Image& Img,
				   const unsigned int& sampNum)
  {
    float2 MINO = make_float2(
			      -Img.m_Size.x / 2.0f + Img.m_Bias.x,
			      -Img.m_Size.y / 2.0f + Img.m_Bias.y);

    float curAng = 0;
    float cosT = 0;
    float sinT = 0;
    Ray2D ray;

    //unsigned int detId;
    float ang(0); //bias angle from the center of the Fan Beams

    float2 curDetPos; //the current detector element position;
    //float totWeight(0);

    float smallDetStep = FanGeo.m_DetStp / sampNum; //\CF²\C9\D1\F9\BA\F3\B5\C4detector\B2\BD\B3\A4;
    float cntSmallDet = sampNum * 0.5f;
    float realAng = 0;
    unsigned int angIdx = 0;
    unsigned int detIdx = 0;
    unsigned int subDetIdx = 0;
    const unsigned int colNum = Img.m_Reso.x * Img.m_Reso.y;
    int rowidx = 0;
    for (angIdx = 0; angIdx != FanGeo.m_ViwN; ++angIdx)
      {
	//Current rotation angle;
	curAng = FanGeo.m_ViwBeg + FanGeo.m_ViwStp * angIdx;
	cosT = cosf(curAng);
	sinT = sinf(curAng);
	ray.o = rotation(make_float2(0, FanGeo.m_S2O), cosT, sinT);

	for (detIdx = 0; detIdx != FanGeo.m_DetN; ++detIdx)
	  {

	    rowidx = angIdx * FanGeo.m_DetN + detIdx;

	    //Calculate current Angle
	    ang = ((float) detIdx - FanGeo.m_DetCntIdx + 0.5f) * FanGeo.m_DetStp;

	    for (subDetIdx = 0; subDetIdx != sampNum; ++subDetIdx)
	      {
		//correct ang
		realAng = ang + (static_cast<float>(subDetIdx) -cntSmallDet + 0.5f) *smallDetStep;
		// current detector element position;
		curDetPos = rotation(make_float2(sinf(realAng) * FanGeo.m_S2D, -cosf(realAng) * FanGeo.m_S2D + FanGeo.m_S2O), cosT, sinT);

		// X-ray direction
		ray.d = normalize(curDetPos - ray.o);

		pushMatrix_Direct(A,ray.o.x, ray.o.y,
				  curDetPos.x, curDetPos.y,
				  MINO.x, MINO.y,
				  Img.m_Step.x, Img.m_Step.y,
				  Img.m_Reso.x, Img.m_Reso.y,
				  rowidx, colNum, sampNum);

	      }
	  }
      }

  }

  void matlabCodeGenerationForReadMatrix(int rowNum, int colNum){
    std::string scripts = "fid = fopen('MatrixA" + std::to_string(rowNum) + "x" + std::to_string(colNum) + ".matrix');\n";
    scripts = scripts + "A = fread(fid," + std::to_string(rowNum) + "*" + std::to_string(colNum) + ",'float');\n";
    scripts = scripts + "A = reshape(A, " + std::to_string(colNum) + ", " + std::to_string(rowNum) + ");\n";
    scripts = scripts + "A = A';\n";

    std::cout<<scripts;
    std::ofstream fou("readGeneratedMatrix.m", std::ios::out|std::ios::app);
    fou<<scripts;
    fou.close();
  }

  void genMatrixAndCalculateSVD()
  {
    FanEAGeo FanGeo;
    Image Img;
    std::ifstream fin("configurationFile.txt", std::ios::in);
    assert(fin.is_open()); // need to use the file
		
    std::string s;
    double val[14];
    int i = 0;
    while (fin >> s)
      {
	if ((i % 2))
	  {
	    std::stringstream ss;
	    ss << s;
	    ss >> val[i / 2];
	    std::cout << val[i / 2] << std::endl;
	  }
	++i;
      }
    FanGeo.m_S2O = val[0];
    FanGeo.m_O2D = val[1];
    FanGeo.m_ViwN = val[2];
    FanGeo.m_ViwBeg = val[3];
    FanGeo.m_ViwStp = val[4];
    FanGeo.m_DetArc = val[5];
    FanGeo.m_DetN = val[6];
    FanGeo.m_DetCntIdx = val[7];

    FanGeo.m_DetStp = val[5] / val[6];
    FanGeo.m_S2D = val[0] + val[1];
    Img.m_Size.x = val[8];
    Img.m_Size.y = val[9];
    Img.m_Reso.x = val[10];
    Img.m_Reso.y = val[11];
    Img.m_Bias.x = val[12];
    Img.m_Bias.y = val[13];
    Img.m_Step.x = Img.m_Size.x / Img.m_Reso.x;
    Img.m_Step.y = Img.m_Size.y / Img.m_Reso.y;



    std::stringstream nonZ;
    float S2O = val[0];
    float O2D = val[1];
    float detCntIdx = val[7];
    float detStp = val[5] / val[6];
    int DN = val[6];
    float ObjStpx = val[8] / val[10];
    float ObjStpy = val[9] / val[11];
    int XN = val[10];
    int YN = val[11];
    float objCntIdxx = (XN - 1.0) * 0.5;
    float objCntIdxy = (YN - 1.0) * 0.5;
    int PN = val[2];
    std::vector<float> angs(PN, 0);
    for (unsigned int i = 0; i != PN; ++i)
      {
	angs[i] = val[3] + i * val[4];
      }

    const int rowNum = FanGeo.m_ViwN * FanGeo.m_DetN;
    const int colNum = Img.m_Reso.x*Img.m_Reso.y;

    float* A = new float[rowNum*colNum];
    for(int i = 0; i != rowNum * colNum; ++i) {
      A[i] = 0;
    }

    genProj_SIDDON_DirectMatrix(A, FanGeo, Img, 16); // Generate the matrix A
		
    std::string AF = "MatrixA" + std::to_string(rowNum)+"x"+std::to_string(colNum)+".matrix"; // Save the matrix A
    std::ofstream Afile(AF.c_str(), std::ios::binary);
    Afile.write((char*) &(A[0]), sizeof(float) * rowNum * colNum);
    Afile.close();

    float* U = new float[rowNum * rowNum];
    float* Sigma = new float[std::min(rowNum, colNum)];
    float* Vt = new float[colNum * colNum];
    int info = calculateSVD(rowNum, colNum, A, U, Sigma, Vt); //Calculate the SVD
    matlabCodeGenerationForReadMatrix(rowNum, colNum);
    for(int i = 0; i != 10; ++i) {
      std::cout<<Sigma[i]<<"  ";
    }

    std::string UF = "MatrixU" + std::to_string(rowNum)+"x"+ std::to_string(rowNum)+".matrix";
    std::ofstream Ufile(UF.c_str(), std::ios::binary);
    Ufile.write((char*) &(U[0]), sizeof(float) * rowNum * rowNum);
    Ufile.close();

    std::string VTF = "MatrixVt" + std::to_string(colNum)+"x"+ std::to_string(colNum)+".matrix";
    std::ofstream VTfile(VTF.c_str(), std::ios::binary);
    VTfile.write((char*) &(Vt[0]), sizeof(float) * colNum * colNum);
    VTfile.close();

    std::string SigF = "MatrixSig" + std::to_string(std::min(rowNum, colNum)) + ".matrix";
    std::ofstream Sigfile(SigF.c_str(), std::ios::binary);
    Sigfile.write((char*) &(SigF[0]), sizeof(float) * std::min(rowNum, colNum));
    Sigfile.close();

		
    delete[] A;
    delete[] U;
    delete[] Sigma;
    delete[] Vt;
  }


};


int main(){
  genSysMatrix::genMatrixAndCalculateSVD();
  return 0;
}
