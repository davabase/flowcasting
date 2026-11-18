// Native CUDA implementation; see README.md for solver differences and references.
#include "flowcast.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
thread_local std::string error;
void check(cudaError_t code) {
    if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}
#define CU(x) check(x)
constexpr int threads = 256;
struct Context {
    int device;
    std::string name;
    cudaStream_t stream{};
    cudaEvent_t events[5]{};
    char* memory = nullptr;
    size_t capacity = 0, used = 0;
    explicit Context(int d) : device(d) {
        CU(cudaSetDevice(device));
        cudaDeviceProp prop{}; CU(cudaGetDeviceProperties(&prop, device)); name = prop.name;
        CU(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        for (auto& e : events) CU(cudaEventCreate(&e));
    }
    ~Context() {
        cudaSetDevice(device);
        if (stream) cudaStreamSynchronize(stream);
        if (memory) cudaFree(memory);
        for (auto e : events) if (e) cudaEventDestroy(e);
        if (stream) cudaStreamDestroy(stream);
    }
    void reset(size_t bytes) {
        CU(cudaSetDevice(device));
        if (bytes > capacity) {
            if (memory) CU(cudaFree(memory)); memory = nullptr; capacity = 0;
            CU(cudaMalloc(&memory, bytes)); capacity = bytes;
        }
        used = 0;
    }
    template<class T> T* alloc(size_t n) {
        size_t start = (used + 255) & ~size_t(255);
        used = start + n * sizeof(T);
        if (used > capacity) throw std::runtime_error("CUDA workspace exhausted");
        return reinterpret_cast<T*>(memory + start);
    }
};
__device__ int clampi(int x, int hi) { return max(0, min(x, hi-1)); }
__device__ int reflect(int x, int hi) {
    if (hi == 1) return 0;
    while (x < 0 || x >= hi) x = x < 0 ? -x : 2*hi-2-x;
    return x;
}
__device__ float read(const float* a, int x, int y, int w, int h, int c=0, int channels=1, bool zero=false) {
    if (zero && (x < 0 || x >= w || y < 0 || y >= h)) return 0;
    return a[(clampi(y,h)*w+clampi(x,w))*channels+c];
}
// OpenCV remap rounds coordinates to a 1/32-pixel interpolation table.
__device__ float sample(const float* a, float x, float y, int w, int h, int c=0,
                        int channels=1, bool zero=false, bool quantize=true) {
    if (quantize) { x = nearbyintf(x*32.f)/32.f; y = nearbyintf(y*32.f)/32.f; }
    // Avoid integer overflow in sampling while preserving constant-border behavior.
    x = fminf(fmaxf(x,-32766.f),32766.f); y = fminf(fmaxf(y,-32766.f),32766.f);
    int ix = int(floorf(x)), iy = int(floorf(y));
    float fx = x-ix, fy = y-iy;
    return (1-fy)*((1-fx)*read(a,ix,iy,w,h,c,channels,zero)+fx*read(a,ix+1,iy,w,h,c,channels,zero))
         + fy*((1-fx)*read(a,ix,iy+1,w,h,c,channels,zero)+fx*read(a,ix+1,iy+1,w,h,c,channels,zero));
}
__device__ bool inside(float x, float y, int w, int h) { return x>=0 && x<=w-1 && y>=0 && y<=h-1; }
__global__ void convert(const uint8_t* in, float* out, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) out[i]=in[i];
}
__global__ void blur_pass(const float* in, float* out, int w, int h, int channels,
                          int count, float sigma, int radius, bool horizontal) {
    int i=blockIdx.x*blockDim.x+threadIdx.x, n=w*h*channels*count;
    if(i>=n) return;
    int c=i%channels, p=(i/channels)%(w*h), x=p%w, y=p/w;
    const float* a=in+(i/(w*h*channels))*w*h*channels;
    float sum=0, norm=0;
    for(int k=-radius;k<=radius;k++) {
        float weight=expf(-float(k*k)/(2*sigma*sigma));
        sum+=weight*read(a,x+(horizontal?k:0),y+(horizontal?0:k),w,h,c,channels); norm+=weight;
    }
    out[i]=sum/norm;
}
float* blur(Context& ctx, const float* in, int w, int h, int channels, int count, float sigma) {
    int n=w*h*channels*count;
    auto tmp=ctx.alloc<float>(n); auto out=ctx.alloc<float>(n);
    int size=int(std::nearbyint(sigma*8+1))|1, radius=size/2;
    blur_pass<<<(n+255)/256,256,0,ctx.stream>>>(in,tmp,w,h,channels,count,sigma,radius,true);
    blur_pass<<<(n+255)/256,256,0,ctx.stream>>>(tmp,out,w,h,channels,count,sigma,radius,false);
    return out;
}
__global__ void downsample(const float* in, float* out, int iw, int ih, int w, int h, int count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=w*h*count) return;
    int p=i%(w*h), x=p%w, y=p/w, image=i/(w*h);
    const float* a=in+image*iw*ih;
    // Pyramid dimensions are floor-halved; area sampling also handles odd dimensions.
    float l=float(x)*iw/w, r=float(x+1)*iw/w, t=float(y)*ih/h, b=float(y+1)*ih/h, sum=0;
    for(int yy=int(t); yy<int(ceilf(b)); yy++) for(int xx=int(l); xx<int(ceilf(r)); xx++)
        sum+=read(a,xx,yy,iw,ih)*fmaxf(0.f,fminf(r,xx+1.f)-fmaxf(l,float(xx)))
                                       *fmaxf(0.f,fminf(b,yy+1.f)-fmaxf(t,float(yy)));
    out[i]=nearbyintf(sum/((r-l)*(b-t)));
}
__global__ void gradient(const float* images, float* grad, int w, int h, int count, float scale, bool texture) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=w*h*count) return;
    int p=i%(w*h), x=p%w, y=p/w;
    const float* a=images+(i/(w*h))*w*h;
    float gx=0, gy=0;
    for(int k=-1;k<=1;k++) {
        int s=k==0?2:1;
        gx+=s*(a[reflect(y+k,h)*w+reflect(x+1,w)]-a[reflect(y+k,h)*w+reflect(x-1,w)]);
        gy+=s*(a[reflect(y+1,h)*w+reflect(x+k,w)]-a[reflect(y-1,h)*w+reflect(x+k,w)]);
    }
    gx*=scale; gy*=scale;
    if(texture) grad[i]=gx*gx+gy*gy;
    else {grad[2*i]=gx;grad[2*i+1]=gy;}
}
__global__ void resize_flow(const float* in, float* out, int iw, int ih, int w, int h, int count, float scale) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=w*h*count) return;
    int p=i%(w*h), x=p%w, y=p/w;
    const float* a=in+(i/(w*h))*iw*ih*2;
    float sx=(x+.5f)*iw/w-.5f, sy=(y+.5f)*ih/h-.5f;
    out[2*i]=scale*sample(a,sx,sy,iw,ih,0,2,false,false);
    out[2*i+1]=scale*sample(a,sx,sy,iw,ih,1,2,false,false);
}
__device__ float reduce(float v) {
    for(int d=16;d;d/=2) v+=__shfl_down_sync(0xffffffff,v,d);
    return __shfl_sync(0xffffffff,v,0);
}
__device__ float patch_error(const float* a, const float* b, int x, int y, float u, float v,
                             int lane, int w, int h) {
    float d=0, ss=0;
    for(int k=lane;k<64;k+=32) {
        float e=sample(b,x+k%8+u,y+k/8+v,w,h,0,1,false,false)-a[(y+k/8)*w+x+k%8];
        d+=e;ss+=e*e;
    }
    d=reduce(d);ss=reduce(ss); return ss-d*d/64;
}
// One warp per 8x8 patch. Double-buffered spatial propagation avoids scan-order races.
__global__ void inverse_search(const float* images, const float* gradients, const int2* pairs,
                               const float* dense, const float* previous, float* sparse,
                               int w, int h, int ws, int hs, int count, int pass) {
    int lane=threadIdx.x%32, patch=(blockIdx.x*blockDim.x+threadIdx.x)/32;
    if(patch>=ws*hs*count) return;
    int pair=patch/(ws*hs), p=patch%(ws*hs), px=p%ws, py=p/ws, x=px*3, y=py*3;
    const float* a=images+pairs[pair].x*w*h; const float* b=images+pairs[pair].y*w*h;
    const float* g=gradients+pairs[pair].x*w*h*2;
    float u=pass?previous[patch*2]:dense[(pair*w*h+(y+4)*w+x+4)*2];
    float v=pass?previous[patch*2+1]:dense[(pair*w*h+(y+4)*w+x+4)*2+1];
    float best=patch_error(a,b,x,y,u,v,lane,w,h);
    if(pass) for(int direction=0;direction<4;direction++) {
        int nx=px+(direction==0?-1:direction==1?1:0), ny=py+(direction==2?-1:direction==3?1:0);
        if(nx>=0 && nx<ws && ny>=0 && ny<hs) {
            int j=(pair*hs+ny)*ws+nx; float cu=previous[2*j],cv=previous[2*j+1];
            float e=patch_error(a,b,x,y,cu,cv,lane,w,h);
            if(e<best) {best=e;u=cu;v=cv;}
        }
    }
    float ax=0,ay=0,xx=0,xy=0,yy=0;
    float av[2],gx[2],gy[2];
    for(int k=0;k<2;k++) {
        int j=(y+(lane+32*k)/8)*w+x+(lane+32*k)%8;
        av[k]=a[j];gx[k]=g[2*j];gy[k]=g[2*j+1];
        ax+=gx[k];ay+=gy[k];xx+=gx[k]*gx[k];xy+=gx[k]*gy[k];yy+=gy[k]*gy[k];
    }
    ax=reduce(ax);ay=reduce(ay);xx=reduce(xx);xy=reduce(xy);yy=reduce(yy);
    float det=xx*yy-xy*xy; if(fabsf(det)<.001f) det=.001f;
    float startu=u,startv=v,prev=1e30f;
    for(int iter=0;iter<6;iter++) {
        float sd=0,ss=0,dx=0,dy=0;
        for(int k=0;k<2;k++) {
            int j=lane+32*k;
            float d=sample(b,x+j%8+u,y+j/8+v,w,h,0,1,false,false)-av[k];
            sd+=d;ss+=d*d;dx+=gx[k]*d;dy+=gy[k]*d;
        }
        sd=reduce(sd);ss=reduce(ss)-sd*sd/64;
        dx=reduce(dx)-sd*ax/64;dy=reduce(dy)-sd*ay/64;
        u-=(yy*dx-xy*dy)/det;v-=(xx*dy-xy*dx)/det;
        if(ss>=prev) break;prev=ss;
    }
    if(!isfinite(u)||!isfinite(v)||(u-startu)*(u-startu)+(v-startv)*(v-startv)>64) {u=startu;v=startv;}
    if(lane==0) {sparse[2*patch]=u;sparse[2*patch+1]=v;}
}
__global__ void densify(const float* images, const int2* pairs, const float* sparse, float* dense,
                        int w, int h, int ws, int hs, int count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=w*h*count) return;
    int p=i%(w*h), x=p%w,y=p/w,pair=i/(w*h);
    const float* a=images+pairs[pair].x*w*h; const float* b=images+pairs[pair].y*w*h;
    int xl=min(max(0,(x-7+2)/3),ws-1),xr=min(x/3,ws-1);
    int yl=min(max(0,(y-7+2)/3),hs-1),yr=min(y/3,hs-1);
    float u=0,v=0,norm=0;
    for(int yy=yl;yy<=yr;yy++) for(int xx=xl;xx<=xr;xx++) {
        int j=(pair*hs+yy)*ws+xx;float su=sparse[2*j],sv=sparse[2*j+1];
        float weight=1/fmaxf(1.f,fabsf(sample(b,x+su,y+sv,w,h,0,1,false,false)-a[p]));
        u+=weight*su;v+=weight*sv;norm+=weight;
    }
    dense[2*i]=u/fmaxf(norm,1e-6f);dense[2*i+1]=v/fmaxf(norm,1e-6f);
}
// Robust brightness/gradient constancy with red-black relaxation at every pyramid level.
__global__ void refinement_derivatives(const float* images, const int2* pairs, const float* flow,
                                      float* avg, float* dt, int w, int h, int count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=w*h*count) return;
    int p=i%(w*h),pair=i/(w*h);const float* a=images+pairs[pair].x*w*h;
    const float* b=images+pairs[pair].y*w*h;
    float warped=sample(b,p%w+flow[2*i],p/w+flow[2*i+1],w,h);
    avg[i]=.5f*(a[p]+warped);dt[i]=warped-a[p];
}
__device__ float derivative(const float* a,int x,int y,int w,int h,bool vertical) {
    return (-read(a,x+(vertical?0:2),y+(vertical?2:0),w,h)
            +8*read(a,x+(vertical?0:1),y+(vertical?1:0),w,h)
            -8*read(a,x-(vertical?0:1),y-(vertical?1:0),w,h)
            +read(a,x-(vertical?0:2),y-(vertical?2:0),w,h))/12;
}
__global__ void refinement_system(const float* avg,const float* dt,const float* base,const float* delta,
                                  float* coeff,float* weights,int w,int h,int count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=w*h*count)return;
    int p=i%(w*h),x=p%w,y=p/w;const float* a=avg+(i/(w*h))*w*h;
    const float* t=dt+(i/(w*h))*w*h;
    float ix=derivative(a,x,y,w,h,false),iy=derivative(a,x,y,w,h,true),iz=t[p];
    float ixx=(read(a,x+1,y,w,h)-2*a[p]+read(a,x-1,y,w,h));
    float iyy=(read(a,x,y+1,w,h)-2*a[p]+read(a,x,y-1,w,h));
    float ixy=(read(a,x+1,y+1,w,h)-read(a,x+1,y-1,w,h)-read(a,x-1,y+1,w,h)+read(a,x-1,y-1,w,h))*.25f;
    float ixz=derivative(t,x,y,w,h,false),iyz=derivative(t,x,y,w,h,true);
    float u=delta[2*i],v=delta[2*i+1],n=ix*ix+iy*iy+.01f;
    float residual=iz+ix*u+iy*v;
    float weight=2.5f/(sqrtf(residual*residual/n+.0001f)*n);
    float a11=weight*ix*ix,a12=weight*ix*iy,a22=weight*iy*iy,b1=-weight*iz*ix,b2=-weight*iz*iy;
    float n1=ixx*ixx+ixy*ixy+.01f,n2=ixy*ixy+iyy*iyy+.01f;
    float r1=ixz+ixx*u+ixy*v,r2=iyz+ixy*u+iyy*v;
    weight=5.f/sqrtf(r1*r1/n1+r2*r2/n2+.0001f);
    a11+=weight*(ixx*ixx/n1+ixy*ixy/n2);a12+=weight*(ixx*ixy/n1+ixy*iyy/n2);
    a22+=weight*(ixy*ixy/n1+iyy*iyy/n2);b1-=weight*(ixx*ixz/n1+ixy*iyz/n2);b2-=weight*(ixy*ixz/n1+iyy*iyz/n2);
    coeff[5*i]=a11;coeff[5*i+1]=a12;coeff[5*i+2]=a22;coeff[5*i+3]=b1;coeff[5*i+4]=b2;
    int right=i+(x<w-1?1:0),down=i+(y<h-1?w:0);
    float ux=(base[2*right]+delta[2*right])-(base[2*i]+u),vx=(base[2*right+1]+delta[2*right+1])-(base[2*i+1]+v);
    float uy=(base[2*down]+delta[2*down])-(base[2*i]+u),vy=(base[2*down+1]+delta[2*down+1])-(base[2*i+1]+v);
    weights[i]=10.f/sqrtf(ux*ux+vx*vx+uy*uy+vy*vy+.0001f);
}
__global__ void relax(const float* base,float* delta,const float* coeff,const float* weights,int w,int h,int count,int color) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=w*h*count)return;
    int p=i%(w*h),x=p%w,y=p/w;if((x+y)%2!=color)return;
    float a=coeff[5*i],b=coeff[5*i+1],c=coeff[5*i+2],r=coeff[5*i+3],s=coeff[5*i+4],norm=0;
    for(int k=0;k<4;k++) {
        int xx=x+(k==0?-1:k==1?1:0), yy=y+(k==2?-1:k==3?1:0);
        if(xx<0||xx>=w||yy<0||yy>=h)continue;
        int j=i+(xx-x)+(yy-y)*w;
        float wt=(weights[i]+weights[j])*.5f;norm+=wt;
        r+=wt*(base[2*j]-base[2*i]+delta[2*j]);s+=wt*(base[2*j+1]-base[2*i+1]+delta[2*j+1]);
    }
    float u=(r-b*delta[2*i+1])/fmaxf(a+norm,1e-6f);
    u=delta[2*i]+1.6f*(u-delta[2*i]);
    float v=(s-b*u)/fmaxf(c+norm,1e-6f);v=delta[2*i+1]+1.6f*(v-delta[2*i+1]);
    delta[2*i]=isfinite(u)?u:0;delta[2*i+1]=isfinite(v)?v:0;
}
__global__ void add_delta(float* flow,const float* delta,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)flow[i]+=delta[i];
}
float* dis(Context& ctx,const float* frames,int t,int w,int h,const std::vector<int2>& host_pairs) {
    int count=int(host_pairs.size()),coarse=std::min(int(std::log2(std::max(w,h)/32.)+.5),int(std::log2(std::min(w,h)/8.)));
    coarse=std::max(0,coarse);int fine=std::min(1,coarse);
    auto pairs=ctx.alloc<int2>(count);CU(cudaMemcpyAsync(pairs,host_pairs.data(),count*sizeof(int2),cudaMemcpyHostToDevice,ctx.stream));
    std::vector<float*> pyramid(coarse+1);std::vector<int> widths(coarse+1),heights(coarse+1);
    pyramid[0]=const_cast<float*>(frames);widths[0]=w;heights[0]=h;
    for(int l=1;l<=coarse;l++) {
        int lw=widths[l]=w>>l,lh=heights[l]=h>>l,n=lw*lh*t;
        pyramid[l]=ctx.alloc<float>(n);
        downsample<<<(n+255)/256,256,0,ctx.stream>>>(pyramid[l-1],pyramid[l],widths[l-1],heights[l-1],lw,lh,t);
    }
    float* flow=nullptr;int oldw=0,oldh=0;
    for(int l=coarse;l>=fine;l--) {
        int lw=widths[l],lh=heights[l],n=lw*lh*count,ws=1+(lw-8)/3,hs=1+(lh-8)/3,np=ws*hs*count;
        auto dense=ctx.alloc<float>(2*n);
        if(flow) resize_flow<<<(n+255)/256,256,0,ctx.stream>>>(flow,dense,oldw,oldh,lw,lh,count,2);
        else CU(cudaMemsetAsync(dense,0,2*n*sizeof(float),ctx.stream));
        auto grad=ctx.alloc<float>(lw*lh*t*2);
        gradient<<<(lw*lh*t+255)/256,256,0,ctx.stream>>>(pyramid[l],grad,lw,lh,t,1,false);
        auto s0=ctx.alloc<float>(np*2),s1=ctx.alloc<float>(np*2);
        for(int pass=0;pass<4;pass++) {
            inverse_search<<<(np*32+127)/128,128,0,ctx.stream>>>(pyramid[l],grad,pairs,dense,s0,s1,lw,lh,ws,hs,count,pass);
            std::swap(s0,s1);
        }
        densify<<<(n+255)/256,256,0,ctx.stream>>>(pyramid[l],pairs,s0,dense,lw,lh,ws,hs,count);
        auto avg=ctx.alloc<float>(n),dt=ctx.alloc<float>(n),delta=ctx.alloc<float>(2*n),coeff=ctx.alloc<float>(5*n),weights=ctx.alloc<float>(n);
        refinement_derivatives<<<(n+255)/256,256,0,ctx.stream>>>(pyramid[l],pairs,dense,avg,dt,lw,lh,count);
        CU(cudaMemsetAsync(delta,0,2*n*sizeof(float),ctx.stream));
        for(int iter=0;iter<5;iter++) {
            refinement_system<<<(n+255)/256,256,0,ctx.stream>>>(avg,dt,dense,delta,coeff,weights,lw,lh,count);
            for(int k=0;k<5;k++) for(int color=0;color<2;color++)
                relax<<<(n+255)/256,256,0,ctx.stream>>>(dense,delta,coeff,weights,lw,lh,count,color);
        }
        add_delta<<<(2*n+255)/256,256,0,ctx.stream>>>(dense,delta,2*n);
        flow=dense;oldw=lw;oldh=lh;
    }
    auto full=ctx.alloc<float>(w*h*count*2);
    resize_flow<<<(w*h*count+255)/256,256,0,ctx.stream>>>(flow,full,oldw,oldh,w,h,count,float(1<<fine));
    return full;
}

struct Pair { int current,previous,gap; };
__global__ void pair_fields(const float* frames,const float* flows,const float* texture,const Pair* pairs,
                            float* confidence,float* residual,float* valid,int w,int h,int count,FCConfig cfg) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=w*h*count)return;
    int p=i%(w*h),j=i/(w*h),x=p%w,y=p/w;
    Pair pair=pairs[j];const float* back=flows+j*w*h*4;const float* forward=back+w*h*2;
    float u=back[2*p],v=back[2*p+1],sx=x+u,sy=y+v;
    float cur=frames[pair.current*w*h+p],prev=sample(frames+pair.previous*w*h,sx,sy,w,h);
    float e=(cur-prev)/cfg.photometric_scale;
    float cx=u+sample(forward,sx,sy,w,h,0,2),cy=v+sample(forward,sx,sy,w,h,1,2);
    float consistency=cfg.consistency_scale*pair.gap,tex=sqrtf(fmaxf(0,texture[pair.current*w*h+p]));
    confidence[i]=cfg.confidence_weighting?(inside(sx,sy,w,h)/(1+e*e)/(1+(cx*cx+cy*cy)/(consistency*consistency))*(.05f+.95f*tex/(tex+3))):1;
    residual[i]=(cur-prev)/pair.gap;valid[i]=inside(sx,sy,w,h)?1:0;
}
__global__ void align_history(const float* flows,const float* confidence,const float* residual,const float* valid,
                              float* aligned,float* weights,float* trends,float* masks,int w,int h,int start,int count,int gap,FCConfig cfg) {
    int p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=w*h)return;
    float x=float(p%w),y=float(p/w);bool available=true;
    for(int age=0;age<count;age++) {
        int j=start+age;const float* flow=flows+j*w*h*4;
        float u=sample(flow,x,y,w,h,0,2)/gap,v=sample(flow,x,y,w,h,1,2)/gap;
        bool ok=available&&inside(x,y,w,h)&&sample(valid+j*w*h,x,y,w,h)>=.999f;
        aligned[(j*w*h+p)*2]=cfg.align_motion?u:flow[2*p]/gap;
        aligned[(j*w*h+p)*2+1]=cfg.align_motion?v:flow[2*p+1]/gap;
        weights[j*w*h+p]=(cfg.align_motion?sample(confidence+j*w*h,x,y,w,h)*ok:confidence[j*w*h+p])*powf(cfg.recency_decay,float(age));
        trends[j*w*h+p]=sample(residual+j*w*h,x,y,w,h);masks[j*w*h+p]=ok?1:0;
        available=ok;x+=u;y+=v;
    }
}
__device__ float median(float* a,int n) {
    if(!n)return 0;
    for(int i=1;i<n;i++){float v=a[i];int j=i;while(j>0&&a[j-1]>v){a[j]=a[j-1];j--;}a[j]=v;}
    return n%2?a[n/2]:.5f*(a[n/2-1]+a[n/2]);
}
// Pack mean*reliability (xy), reliability and weighted fallback (xy) for filtering.
__global__ void combine(const float* aligned,const float* weights,const float* trends,const float* masks,
                        float* packed,float* raw,float* growth,int w,int h,int start,int count,FCConfig cfg) {
    int p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=w*h)return;
    float u=0,v=0,fx=0,fy=0,support=0,recency=0,a[8];int used=0;
    for(int age=0;age<count;age++) {
        int i=(start+age)*w*h+p;float r=powf(cfg.recency_decay,float(age)),weight=weights[i];
        fx+=r*aligned[2*i];fy+=r*aligned[2*i+1];recency+=r;
        u+=weight*aligned[2*i];v+=weight*aligned[2*i+1];support+=weight;
        if(masks[i])a[used++]=trends[i];
    }
    fx/=recency;fy/=recency;
    u=support>1e-6f?u/fmaxf(support,1e-6f):fx;v=support>1e-6f?v/fmaxf(support,1e-6f):fy;
    float reliability=support/recency;packed[3*p]=u*reliability;packed[3*p+1]=v*reliability;packed[3*p+2]=reliability;
    raw[4*p]=u;raw[4*p+1]=v;raw[4*p+2]=fx;raw[4*p+3]=fy;
    float coverage=float(used)/count;growth[2*p]=median(a,used)*coverage;growth[2*p+1]=coverage;
}
__global__ void finish_gap(const float* packed,const float* smooth,const float* raw,const float* growth,
                           float* motion,float* trend,int n,FCConfig cfg) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    float trust=fminf(1.f,packed[3*i+2]/.5f),norm=smooth[3*i+2];
    for(int c=0;c<2;c++) {
        float sm=norm>1e-6f?smooth[3*i+c]/fmaxf(norm,1e-6f):raw[4*i+2+c];
        motion[2*i+c]=cfg.smooth_motion?trust*raw[4*i+c]+(1-trust)*sm:raw[4*i+c];
    }
    trend[i]=fminf(cfg.growth_cap,fmaxf(-cfg.growth_cap,growth[2*i]/fmaxf(growth[2*i+1],1e-6f)));
}
__global__ void fuse(const float* motions,const float* trends,float* motion,float* trend,int n,int count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    float u=0,v=0,a[8];
    for(int g=0;g<count;g++){u+=motions[(g*n+i)*2];v+=motions[(g*n+i)*2+1];a[g]=trends[g*n+i];}
    motion[2*i]=u/count;motion[2*i+1]=v/count;trend[i]=median(a,count);
}
__global__ void difference(const float* a,const float* b,float* out,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=a[i]-b[i];
}
struct Bands {const float* coarse;const float* detail[8];};
__global__ void advect(const float* last,const float* motion,const float* trend,Bands bands,
                       uint8_t* output,int w,int h,int steps,FCConfig cfg) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=w*h)return;
    float x=float(i%w),y=float(i/w);
    for(int k=1;k<=steps;k++) {
        float dx=sample(motion,x,y,w,h,0,2),dy=sample(motion,x,y,w,h,1,2);
        if(cfg.midpoint) {float mx=x+.5f*dx,my=y+.5f*dy;dx=sample(motion,mx,my,w,h,0,2);dy=sample(motion,mx,my,w,h,1,2);}
        x+=dx;y+=dy;
        float value=sample(bands.coarse,x,y,w,h,0,1,cfg.zero_boundary);
        for(int b=0;b<(cfg.multiscale?cfg.scale_count:0);b++) value+=expf(-k/cfg.lifetimes[b])*sample(bands.detail[b],x,y,w,h,0,1,cfg.zero_boundary);
        if(cfg.growth_decay&&trend) value+=cfg.growth_lifetime*(-expm1f(-k/cfg.growth_lifetime))*sample(trend,x,y,w,h,0,1,cfg.zero_boundary);
        output[(k-1)*w*h+i]=uint8_t(fminf(255.f,fmaxf(0.f,nearbyintf(value))));
    }
}
void forecast(Context& ctx,const float* last,const float* motion,const float* trend,uint8_t* out,int w,int h,int steps,FCConfig cfg) {
    Bands bands{};bands.coarse=last;const float* prev=last;
    if(cfg.multiscale)for(int i=0;i<cfg.scale_count;i++) {
        auto coarse=blur(ctx,last,w,h,1,1,cfg.sigmas[i]);auto detail=ctx.alloc<float>(w*h);
        difference<<<(w*h+255)/256,256,0,ctx.stream>>>(prev,coarse,detail,w*h);
        bands.detail[i]=detail;prev=coarse;bands.coarse=coarse;
    }
    advect<<<(w*h+255)/256,256,0,ctx.stream>>>(last,motion,trend,bands,out,w,h,steps,cfg);
}
void validate(int t,int h,int w,int steps,const FCConfig& cfg) {
    if(t<2||h<12||w<12||steps<1||cfg.pairs<1||cfg.pairs>8||cfg.gap_count<1||cfg.gap_count>8||cfg.scale_count<1||cfg.scale_count>8)
        throw std::invalid_argument("Invalid dimensions or CUDA configuration limits (1..8 pairs/gaps/scales)");
}
} // namespace

extern "C" {
void* fc_create(int device) {
    try {error.clear();return new Context(device);} catch(const std::exception& e){error=e.what();return nullptr;}
}
void fc_destroy(void* ptr) {delete static_cast<Context*>(ptr);}
const char* fc_error(){return error.c_str();}
const char* fc_device(void* ptr){return static_cast<Context*>(ptr)->name.c_str();}
int fc_runtime_version(){int v=0;cudaRuntimeGetVersion(&v);return v;}
int fc_predict(void* ptr,const uint8_t* input,int t,int h,int w,int steps,const FCConfig* config,
               const float* supplied,uint8_t* prediction,float* motion,FCTiming* timing) {
    try {
        error.clear();Context& ctx=*static_cast<Context*>(ptr);const auto cfg=*config;validate(t,h,w,steps,cfg);
        std::vector<int> gaps;for(int g=0;g<cfg.gap_count;g++) if(cfg.gaps[g]<t)gaps.push_back(cfg.gaps[g]);
        if(gaps.empty())gaps.push_back(1);
        std::vector<Pair> host_pairs;std::vector<int2> motion_pairs;std::vector<int> starts,counts;
        for(int gap:gaps) {
            starts.push_back(int(host_pairs.size()));counts.push_back(std::min(cfg.pairs,t-gap));
            for(int age=0;age<counts.back();age++) {
                int cur=t-1-age,prev=cur-gap;host_pairs.push_back({cur,prev,gap});
                motion_pairs.push_back(make_int2(cur,prev));motion_pairs.push_back(make_int2(prev,cur));
            }
        }
        int n=w*h,np=int(host_pairs.size()),ng=int(gaps.size());
        ctx.reset(size_t(n)*(512+size_t(np)*320+size_t(t)*32+steps*2)+1024*1024);
        auto bytes=ctx.alloc<uint8_t>(n*t);auto frames=ctx.alloc<float>(n*t);
        auto out=ctx.alloc<uint8_t>(n*steps);auto final_motion=ctx.alloc<float>(n*2),final_trend=ctx.alloc<float>(n);
        CU(cudaEventRecord(ctx.events[0],ctx.stream));
        CU(cudaMemcpyAsync(bytes,input,n*t,cudaMemcpyHostToDevice,ctx.stream));
        convert<<<(n*t+255)/256,256,0,ctx.stream>>>(bytes,frames,n*t);
        float* flows=nullptr;
        if(supplied) {flows=ctx.alloc<float>(np*n*4);CU(cudaMemcpyAsync(flows,supplied,np*n*4*sizeof(float),cudaMemcpyHostToDevice,ctx.stream));}
        CU(cudaEventRecord(ctx.events[1],ctx.stream));
        if(!supplied)flows=dis(ctx,frames,t,w,h,motion_pairs);
        CU(cudaEventRecord(ctx.events[2],ctx.stream));
        auto grad=ctx.alloc<float>(n*t);gradient<<<(n*t+255)/256,256,0,ctx.stream>>>(frames,grad,w,h,t,.125f,true);
        auto texture=blur(ctx,grad,w,h,1,t,2);
        auto pairs=ctx.alloc<Pair>(np);CU(cudaMemcpyAsync(pairs,host_pairs.data(),np*sizeof(Pair),cudaMemcpyHostToDevice,ctx.stream));
        auto confidence=ctx.alloc<float>(n*np),residual=ctx.alloc<float>(n*np),valid=ctx.alloc<float>(n*np);
        pair_fields<<<(n*np+255)/256,256,0,ctx.stream>>>(frames,flows,texture,pairs,confidence,residual,valid,w,h,np,cfg);
        auto aligned=ctx.alloc<float>(n*np*2),weights=ctx.alloc<float>(n*np),trends=ctx.alloc<float>(n*np),masks=ctx.alloc<float>(n*np);
        auto motions=ctx.alloc<float>(ng*n*2),growths=ctx.alloc<float>(ng*n);
        for(int g=0;g<ng;g++) {
            align_history<<<(n+255)/256,256,0,ctx.stream>>>(flows,confidence,residual,valid,aligned,weights,trends,masks,w,h,starts[g],counts[g],gaps[g],cfg);
            auto packed=ctx.alloc<float>(3*n),raw=ctx.alloc<float>(4*n),growth=ctx.alloc<float>(2*n);
            combine<<<(n+255)/256,256,0,ctx.stream>>>(aligned,weights,trends,masks,packed,raw,growth,w,h,starts[g],counts[g],cfg);
            auto smooth=cfg.smooth_motion?blur(ctx,packed,w,h,3,1,cfg.smoothing_sigma):packed;
            auto smgrowth=blur(ctx,growth,w,h,2,1,cfg.growth_sigma);
            finish_gap<<<(n+255)/256,256,0,ctx.stream>>>(packed,smooth,raw,smgrowth,motions+g*n*2,growths+g*n,n,cfg);
        }
        fuse<<<(n+255)/256,256,0,ctx.stream>>>(motions,growths,final_motion,final_trend,n,ng);
        forecast(ctx,frames+(t-1)*n,final_motion,final_trend,out,w,h,steps,cfg);
        CU(cudaEventRecord(ctx.events[3],ctx.stream));
        CU(cudaMemcpyAsync(prediction,out,n*steps,cudaMemcpyDeviceToHost,ctx.stream));
        CU(cudaMemcpyAsync(motion,final_motion,n*2*sizeof(float),cudaMemcpyDeviceToHost,ctx.stream));
        CU(cudaEventRecord(ctx.events[4],ctx.stream));CU(cudaStreamSynchronize(ctx.stream));CU(cudaGetLastError());
        if(timing) {
            CU(cudaEventElapsedTime(&timing->upload_ms,ctx.events[0],ctx.events[1]));
            CU(cudaEventElapsedTime(&timing->motion_ms,ctx.events[1],ctx.events[2]));
            CU(cudaEventElapsedTime(&timing->evolution_ms,ctx.events[2],ctx.events[3]));
            CU(cudaEventElapsedTime(&timing->download_ms,ctx.events[3],ctx.events[4]));
        }
        return 0;
    }catch(const std::exception& e){error=e.what();return -1;}
}
int fc_advect(void* ptr,const float* last,const float* motion,const float* trend,int h,int w,int steps,const FCConfig* config,uint8_t* prediction) {
    try {
        error.clear();Context& ctx=*static_cast<Context*>(ptr);validate(2,h,w,steps,*config);int n=w*h;
        ctx.reset(size_t(n)*(256+steps)+1024*1024);auto a=ctx.alloc<float>(n),m=ctx.alloc<float>(2*n),tr=ctx.alloc<float>(n);auto out=ctx.alloc<uint8_t>(n*steps);
        CU(cudaMemcpyAsync(a,last,n*sizeof(float),cudaMemcpyHostToDevice,ctx.stream));
        CU(cudaMemcpyAsync(m,motion,2*n*sizeof(float),cudaMemcpyHostToDevice,ctx.stream));
        if(trend)CU(cudaMemcpyAsync(tr,trend,n*sizeof(float),cudaMemcpyHostToDevice,ctx.stream));
        forecast(ctx,a,m,trend?tr:nullptr,out,w,h,steps,*config);
        CU(cudaMemcpyAsync(prediction,out,n*steps,cudaMemcpyDeviceToHost,ctx.stream));CU(cudaStreamSynchronize(ctx.stream));CU(cudaGetLastError());return 0;
    }catch(const std::exception& e){error=e.what();return -1;}
}
}
