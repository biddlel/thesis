#include "doa_music.h"
#include <cmath>

#define TWO_PI_F 6.28318530718f

DOAMusic::DOAMusic() : AudioStream(NUM_MICS, _inputQ) {}

void DOAMusic::begin() {
    arm_rfft_fast_init_f32(&_rfft, FFT_SIZE);
}

void DOAMusic::update() {
    audio_block_t* in[NUM_MICS];
    for(int ch=0; ch<NUM_MICS; ++ch) in[ch] = receiveReadOnly(ch);
    if(!in[0]) return;

    const float scale = 1.0f/32768.0f;
    uint16_t off = (_blkCnt % HISTORY_BLOCKS) * BLOCK_SAMPLES;

    for(int ch=0; ch<NUM_MICS; ++ch){
        float *dst = &_hist[ch][off];
        const int16_t *src = in[ch]->data;
        for(int i=0;i<BLOCK_SAMPLES;++i) dst[i] = src[i]*scale;
    }

    ++_blkCnt;
    if(_blkCnt % HISTORY_BLOCKS == 0){
        buildCovariance();
        eigenNoiseSubspace();
        _azimuthDeg = scanAzimuth();
        _conf = 1.0f;
        _new = true;
    }

    for(int ch=0; ch<NUM_MICS; ++ch) release(in[ch]);
}

// ---------- Covariance from freq bins above MIN_FREQ_HZ ----------
void DOAMusic::buildCovariance(){
    const int minBin = (int)ceilf(MIN_FREQ_HZ * FFT_SIZE / SAMPLE_RATE_HZ);
    const int maxBin = FFT_SIZE/2;
    const int bins   = maxBin - minBin;

    static float32_t spec[NUM_MICS][FFT_SIZE]; // complex spectrum

    for(int ch=0; ch<NUM_MICS; ++ch){
        arm_rfft_fast_f32(&_rfft, _hist[ch], spec[ch], 0);
    }

    for(int r=0;r<NUM_MICS;++r){
        for(int c=0;c<NUM_MICS;++c){
            float acc=0;
            for(int k=minBin;k<maxBin;++k){
                int re=2*k, im=2*k+1;
                float xr=spec[r][re], xi=spec[r][im];
                float yr=spec[c][re], yi=-spec[c][im]; // conj
                acc += xr*yr - xi*yi;
            }
            _R[r*NUM_MICS+c] = acc / bins;
        }
    }
}

// ---------- Eigen decomposition ----------
static void jacobi4(float32_t *A, float32_t *V, float32_t *d); // forward

void DOAMusic::eigenNoiseSubspace(){
    jacobi4(_R, _eigVec, _eigVal);
    for(int i=0;i<NUM_MICS-1;++i){
        for(int j=i+1;j<NUM_MICS;++j){
            if(_eigVal[i]>_eigVal[j]){
                std::swap(_eigVal[i],_eigVal[j]);
                for(int k=0;k<NUM_MICS;++k)
                    std::swap(_eigVec[k+i*NUM_MICS], _eigVec[k+j*NUM_MICS]);
            }
        }
    }
}

// ---------- MUSIC scan ----------
float DOAMusic::scanAzimuth(){
    float best=0; int bestIdx=0;
    // noise subspace proj
    float En[NUM_MICS*(NUM_MICS-1)];
    for(int c=0;c<NUM_MICS-1;++c)
        for(int r=0;r<NUM_MICS;++r)
            En[r + c*NUM_MICS] = _eigVec[r + c*NUM_MICS];

    float EnEnH[NUM_MICS*NUM_MICS]={0};
    for(int r=0;r<NUM_MICS;++r)
        for(int c=0;c<NUM_MICS;++c){
            float s=0;
            for(int k=0;k<NUM_MICS-1;++k)
                s += En[r+k*NUM_MICS]*En[c+k*NUM_MICS];
            EnEnH[r*NUM_MICS+c]=s;
        }

    for(int ang=0; ang<360; ++ang){
        float th=ang*DEG2RAD;
        float sv[NUM_MICS];
        for(int m=0;m<NUM_MICS;++m){
            float x=micPos[m][0]*0.001f, y=micPos[m][1]*0.001f;
            float proj = x*cosf(th)+y*sinf(th);
            float delay = proj / SPEED_OF_SOUND;
            sv[m]=cosf(TWO_PI_F*SAMPLE_RATE_HZ*delay);
        }
        float tmp[NUM_MICS]={0};
        for(int c=0;c<NUM_MICS;++c)
            for(int r=0;r<NUM_MICS;++r)
                tmp[c]+=sv[r]*EnEnH[r*NUM_MICS+c];
        float denom=1e-9f;
        for(int i=0;i<NUM_MICS;++i) denom+=tmp[i]*sv[i];
        float P=1.0f/denom;
        if(P>best){best=P; bestIdx=ang;}
    }
    return (float)bestIdx;
}

// ---------- 4x4 Jacobi eigensolver ----------
static void jacobi4(float32_t *A,float32_t *V,float32_t *d){
    for(int i=0;i<4;++i){
        for(int j=0;j<4;++j) V[i*4+j]=(i==j)?1:0;
        d[i]=A[i*4+i];
    }
    float B[4],Z[4]; memcpy(B,d,sizeof(B)); memset(Z,0,sizeof(Z));
    for(int it=0; it<25; ++it){
        float sm=0; for(int p=0;p<3;++p) for(int q=p+1;q<4;++q) sm+=fabsf(A[p*4+q]);
        if(sm<1e-9f) break;
        float tresh = (it<3)?0.2f*sm/16.0f:0.0f;
        for(int p=0;p<3;++p){
            for(int q=p+1;q<4;++q){
                float g=100*fabsf(A[p*4+q]);
                if(it>3 && fabsf(d[p])+g==fabsf(d[p]) && fabsf(d[q])+g==fabsf(d[q]))
                    A[p*4+q]=0;
                else if(fabsf(A[p*4+q])>tresh){
                    float h=d[q]-d[p], t;
                    if(fabsf(h)+g==fabsf(h)) t=A[p*4+q]/h;
                    else{
                        float theta=0.5f*h/A[p*4+q];
                        t=1.0f/(fabsf(theta)+sqrtf(1+theta*theta));
                        if(theta<0) t=-t;
                    }
                    float c=1.0f/sqrtf(1+t*t), s=t*c, tau=s/(1+c);
                    h=t*A[p*4+q]; Z[p]-=h; Z[q]+=h; d[p]-=h; d[q]+=h; A[p*4+q]=0;
                    for(int j=0;j<p;++j){
                        float gA=A[j*4+p], hA=A[j*4+q];
                        A[j*4+p]=gA-s*(hA+gA*tau);
                        A[j*4+q]=hA+s*(gA-hA*tau);
                    }
                    for(int j=p+1;j<q;++j){
                        float gA=A[p*4+j], hA=A[j*4+q];
                        A[p*4+j]=gA-s*(hA+gA*tau);
                        A[j*4+q]=hA+s*(gA-hA*tau);
                    }
                    for(int j=q+1;j<4;++j){
                        float gA=A[p*4+j], hA=A[q*4+j];
                        A[p*4+j]=gA-s*(hA+gA*tau);
                        A[q*4+j]=hA+s*(gA-hA*tau);
                    }
                    for(int j=0;j<4;++j){
                        float gV=V[j*4+p], hV=V[j*4+q];
                        V[j*4+p]=gV-s*(hV+gV*tau);
                        V[j*4+q]=hV+s*(gV-hV*tau);
                    }
                }
            }
        }
        for(int p=0;p<4;++p){ B[p]+=Z[p]; d[p]=B[p]; Z[p]=0; }
    }
}
