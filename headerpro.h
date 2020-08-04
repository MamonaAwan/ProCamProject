#include <opencv\cv.h>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <iostream>
#include <Windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>


using namespace cv;
using namespace std;

Mat Screen, Dst, Src,TestImg, ROIScreen;
Mat Gray, ScrCB, DGray, DBin, Border, Window, SGray;
	
int maxLength, maxLengthROI;
    
vector<Vec2f> linesCS, linesPC, linesRect, linesROI;
vector <Point2f> features;

double alpha=2.2;	//1.0-3.0
int beta=50;		//0-100
Scalar mymean, stddev;
double stdv;

Vec2f topEdgeCS, bottomEdgeCS, leftEdgeCS, rightEdgeCS;
Vec2f topEdgeROI, bottomEdgeROI, leftEdgeROI, rightEdgeROI;
Vec2f topEdgePC, bottomEdgePC, leftEdgePC, rightEdgePC;
Vec2f topEdgeRect, bottomEdgeRect, leftEdgeRect, rightEdgeRect;

CvPoint ptTopLeftCS, ptTopRightCS, ptBotLeftCS, ptBotRightCS;
CvPoint ptTopLeftROI, ptTopRightROI, ptBotLeftROI, ptBotRightROI;
CvPoint ptTopLeftPC, ptTopRightPC, ptBotLeftPC, ptBotRightPC;
CvPoint ptTopLeftRect, ptTopRightRect, ptBotLeftRect, ptBotRightRect;

Point2f src[4], dst[4];
Point2f srcROI[4], dstROI[4];
vector <Point2f> srcpt, dstpt;
Mat C, Undistorted, ROI, ROIImage, RoughROI, T, CROI, UndistortedROI;
Mat P, W, S;
Mat OutBlah, OutReal,OutB, Output;


double MoveX, MoveY;
double WA, WB, HA, HB;
double row, col;
Mat TpLt, TpRt, BtLt, BtRt;
Mat ZTpLt, ZTpRt, ZBtLt, ZBtRt;
Mat ULPt, BLPt, URPt;
Mat St, Su, Sv;

double Size_Width, Size_Height;
int w, h, r, c, Method;
double ScaleX, ScaleY;

void drawLine(Vec2f line, Mat &img, Scalar rgb=CV_RGB(0,0,255));
void mergeRelatedLinesCS(vector<Vec2f>*lines, Mat &img);
void mergeRelatedLinesPC(vector<Vec2f>*lines, Mat &img);
void mergeRelatedLinesROI(vector<Vec2f>*lines, Mat &img);

void drawlines(vector<Vec2f> &lines,Vec2f &topEdge, Vec2f &bottomEdge, Vec2f &rightEdge, Vec2f &leftEdge);
void drawlinesROI(vector<Vec2f> &lines,Vec2f &topEdge, Vec2f &bottomEdge, Vec2f &rightEdge, Vec2f &leftEdge);

void intersectionPoints(Mat &Border,Vec2f &topEdge, Vec2f &bottomEdge, Vec2f &rightEdge, Vec2f &leftEdge, CvPoint &ptTopLeft, CvPoint &ptTopRight, CvPoint &ptBotLeft, CvPoint &ptBotRight);
int findMaxLength(int &maxLength, CvPoint &ptTopLeft, CvPoint &ptTopRight, CvPoint &ptBotLeft, CvPoint &ptBotRight);
void floodFilling(Mat &Img);

void CameraScreen (Mat& Screen, Mat&C);
void ProjectorCamera (Mat &Dst, Mat& Src, Mat &T);
void GetROIinScreen(Mat &Screen, Mat & CROI);

double FindMax(double a, double b, double c, double d);
double FindMin(double a,double b,double c, double d);
bool SameSign(int x, int y);
Mat MakeProjection (Mat &Screen, Mat&Dst, Mat &Src, Mat &TestImage, int Method, double ScaleX, double ScaleY);
void GetDesktopResolution(int& horizontal, int& vertical);
