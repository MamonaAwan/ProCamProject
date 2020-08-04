#include <opencv\cv.h>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;


void drawLine(Vec2f line, Mat &img, Scalar rgb=CV_RGB(0,0,255))
{
 if(line[1]!=0)
 {
	  float m=-1/tan(line[1]);
	  float c=line[0]/sin(line[1]);
	  cv::line(img,Point(0,c),Point(img.size().width,m*img.size().width+c),rgb);
 }
 else
 {
	cv::line(img,Point(line[0],0),Point(line[0],img.size().height),rgb);
 }
}

void mergeRelatedLinesCS(vector<Vec2f>*lines, Mat &img)
{
 vector<Vec2f>::iterator current;
 for( current=lines->begin(); current!=lines->end();current++)
 {
  if((*current)[0]==0 && (*current)[1]==-100)
   continue;
  float p1=(*current)[0];
  float theta1=(*current)[1];
  Point pt1current, pt2current;
  if(theta1>CV_PI*45/180 && theta1<CV_PI*135/180)
  {
	   pt1current.x=0;
	   pt1current.y=p1/sin(theta1);
	   pt2current.x=img.size().width;
	   pt2current.y=-pt2current.x/tan(theta1) + p1/sin(theta1);
  }
  else
  {
	   pt1current.y=0;
	   pt1current.x=p1/cos(theta1);
	   pt2current.y=img.size().height;
	   pt2current.x=-pt2current.y/tan(theta1) + p1/cos(theta1);
  }

  vector<Vec2f>::iterator pos;
  for (pos=lines->begin();pos!=lines->end();pos++)
  {
	  if(*current==*pos)
		continue;
	  if(fabs((*pos)[0]-(*current)[0])<20 && fabs((*pos)[1]-(*current)[1])<CV_PI*10/180)
	  {
			float p=(*pos)[0];
			float theta=(*pos)[1];
			Point pt1, pt2;
			if((*pos)[1]>CV_PI*45/180 && (*pos)[1]<CV_PI*135/180)
			{
				 pt1.x=0;
				 pt1.y=p/sin(theta);
				 pt2.x=img.size().width;
				 pt2.y=-pt2.x/tan(theta) + p/sin(theta);
			}
			else 
			{
				 pt1.y=0;
				 pt1.x=p/cos(theta);
				 pt2.y=img.size().height;
				 pt2.x=-pt2.y/tan(theta) + p/cos(theta);
			}
			if(((double)(pt1.x-pt1current.x)*(pt1.x-pt1current.x) + (pt1.y-pt1current.y)*(pt1.y-pt1current.y)<128*128) &&
			 ((double)(pt2.x-pt2current.x)*(pt2.x-pt2current.x) +(pt2.y-pt2current.y)*(pt2.y-pt2current.y)<128*128))
			{
				 (*current)[0]=((*current)[0]+(*pos)[0])/2;
				 (*current)[1]=((*current)[1]+(*pos)[1])/2;
				 (*pos)[0]=0;
				 (*pos)[1]=-100;
			}
		}
	  }
	 }
}

void mergeRelatedLinesPC(vector<Vec2f>*lines, Mat &img)
{
 vector<Vec2f>::iterator current;
 for( current=lines->begin(); current!=lines->end();current++)
 {
  if((*current)[0]==0 && (*current)[1]==-100)
   continue;
  float p1=(*current)[0];
  float theta1=(*current)[1];
  Point pt1current, pt2current;
  if(theta1>CV_PI*45/180 && theta1<CV_PI*135/180)
  {
	   pt1current.x=0;
	   pt1current.y=p1/sin(theta1);
	   pt2current.x=img.size().width;
	   pt2current.y=-pt2current.x/tan(theta1) + p1/sin(theta1);
  }
  else
  {
	   pt1current.y=0;
	   pt1current.x=p1/cos(theta1);
	   pt2current.y=img.size().height;
	   pt2current.x=-pt2current.y/tan(theta1) + p1/cos(theta1);
  }

  vector<Vec2f>::iterator pos;
  for (pos=lines->begin();pos!=lines->end();pos++)
  {
	  if(*current==*pos)
		continue;
	  if(fabs((*pos)[0]-(*current)[0])<100 && fabs((*pos)[1]-(*current)[1])<CV_PI*30/180)
	  {
			float p=(*pos)[0];
			float theta=(*pos)[1];
			Point pt1, pt2;
			if((*pos)[1]>CV_PI*45/180 && (*pos)[1]<CV_PI*135/180)
			{
				 pt1.x=0;
				 pt1.y=p/sin(theta);
				 pt2.x=img.size().width;
				 pt2.y=-pt2.x/tan(theta) + p/sin(theta);
			}
			else 
			{
				 pt1.y=0;
				 pt1.x=p/cos(theta);
				 pt2.y=img.size().height;
				 pt2.x=-pt2.y/tan(theta) + p/cos(theta);
			}
			if(((double)(pt1.x-pt1current.x)*(pt1.x-pt1current.x) + (pt1.y-pt1current.y)*(pt1.y-pt1current.y)<150*150) &&
			 ((double)(pt2.x-pt2current.x)*(pt2.x-pt2current.x) +(pt2.y-pt2current.y)*(pt2.y-pt2current.y)<150*150))
			{
				 (*current)[0]=((*current)[0]+(*pos)[0])/2;
				 (*current)[1]=((*current)[1]+(*pos)[1])/2;
				 (*pos)[0]=0;
				 (*pos)[1]=-100;
			}
		}
	  }
	 }
}

void drawlines(vector<Vec2f> &lines,Vec2f &topEdge, Vec2f &bottomEdge, Vec2f &rightEdge, Vec2f &leftEdge)
{
	double topYintercept=100000, topXintercept=0; double bottomYintercept=0, bottomXintercept=0;
	double leftXintercept=100000, leftYintercept=0; double rightXintercept=0, rightYintercept=0;
	for(int i=0; i<lines.size(); i++)
	{
		Vec2f current=lines[i];
		float p=current[0];
		float theta=current[1];
		if(p==0 && theta==-100)
			continue;
		double xintercept, yintercept;
		xintercept= p/cos(theta);
		yintercept= p/(cos(theta)*sin(theta));
		if(theta>CV_PI*70/180 && theta<CV_PI*100/180)
		{
			if(p<topEdge[0])
				topEdge=current;
			if(p>bottomEdge[0])
				bottomEdge=current;
		}
		else if (theta<CV_PI*10/180 || theta>CV_PI*170/180)
		{
			if(xintercept>rightXintercept)
			{
				rightEdge=current;
				rightXintercept=xintercept;
			}
			else if(xintercept<=leftXintercept)
			{
				leftEdge=current;
				leftXintercept=xintercept;
			}
		}
	}
}

void intersectionPoints(Mat &Border,Vec2f &topEdge, Vec2f &bottomEdge, Vec2f &rightEdge, Vec2f &leftEdge, CvPoint &ptTopLeft, CvPoint &ptTopRight, CvPoint &ptBotLeft, CvPoint &ptBotRight)
{
	 Point left1,left2, right1,right2, bot1,bot2, top1,top2;
	 int height=Border.size().height;
	 int width=Border.size().width;

	 if(leftEdge[1]!=0)
	 {
		  left1.x=0;  left1.y=leftEdge[0]/sin(leftEdge[1]);
		  left2.x=width; left2.y=-left2.x/tan(leftEdge[1])+left1.y;
	 }
	 else
	 {
		  left1.y=0;   left1.x=leftEdge[0]/cos(leftEdge[1]);
		  left2.y=height; left2.x=left1.x-height*tan(leftEdge[1]);
	 }
	 if(rightEdge[1]!=0)
	 {
		  right1.x=0;     right1.y=rightEdge[0]/sin(rightEdge[1]);
		  right2.x=width; right2.y=-right2.x/tan(rightEdge[1])+right1.y;
	 }
	 else
	 {
		  right1.y=0;      right1.x=rightEdge[0]/cos(rightEdge[1]);
		  right2.y=height; right2.x=right1.x-height*tan(rightEdge[1]);
	 }
	 bot1.x=0; bot1.y=bottomEdge[0]/sin(bottomEdge[1]);
	 bot2.x=width; bot2.y=-bot2.x/tan(bottomEdge[1])+bot1.y;

	 top1.x=0; top1.y=topEdge[0]/sin(topEdge[1]);
	 top2.x=width; top2.y=-top2.x/tan(topEdge[1])+top1.y;

	 double leftA=left2.y-left1.y; double leftB=left1.x-left2.x;
	 double leftC=leftA*left1.x+leftB*left1.y;
 
	 double rightA=right2.y-right1.y; double rightB=right1.x-right2.x;
	 double rightC=rightA*right1.x+rightB*right1.y;

	 double topA=top2.y-top1.y; double topB=top1.x-top2.x;
	 double topC=topA*top1.x+topB*top1.y;

	 double botA=bot2.y-bot1.y; double botB=bot1.x-bot2.x;
	 double botC=botA*bot1.x+botB*bot1.y;

	 double detTopLeft=leftA*topB-leftB*topA;
	 ptTopLeft=cvPoint((topB*leftC-leftB*topC)/detTopLeft,(leftA*topC-topA*leftC)/detTopLeft);
	 double detTopRight=rightA*topB-rightB*topA;
	 ptTopRight=cvPoint((topB*rightC-rightB*topC)/detTopRight,(rightA*topC-topA*rightC)/detTopRight);
 
	 double detBotRight=rightA*botB-rightB*botA;
	 ptBotRight=cvPoint((botB*rightC-rightB*botC)/detBotRight,(rightA*botC-botA*rightC)/detBotRight);
	 double detBotLeft=leftA*botB-leftB*botA;
	 ptBotLeft=cvPoint((botB*leftC-leftB*botC)/detBotLeft,(leftA*botC-botA*leftC)/detBotLeft);
}

int findMaxLength(int &maxLength, CvPoint &ptTopLeft, CvPoint &ptTopRight, CvPoint &ptBotLeft, CvPoint &ptBotRight)
{
	 maxLength =(ptBotLeft.x-ptBotRight.x)*(ptBotLeft.x-ptBotRight.x)+(ptBotLeft.y-ptBotRight.y)*(ptBotLeft.y-ptBotRight.y);
	 int temp= (ptTopRight.x-ptBotRight.x)*(ptTopRight.x-ptBotRight.x)+(ptTopRight.y-ptBotRight.y)*(ptTopRight.y-ptBotRight.y);
	 if(temp>maxLength)
	  maxLength=temp;
	 temp=(ptTopRight.x-ptTopLeft.x)*(ptTopRight.x-ptTopLeft.x)+(ptTopRight.y-ptTopLeft.y)*(ptTopRight.y-ptTopLeft.y);
	 if(temp>maxLength)
	  maxLength=temp;
	 temp=(ptBotLeft.x-ptTopLeft.x)*(ptBotLeft.x-ptTopLeft.x)+(ptBotLeft.y-ptTopLeft.y)*(ptBotLeft.y-ptTopLeft.y);
	 if(temp>maxLength)
	  maxLength=temp;
	 return maxLength=sqrt((double)maxLength);
}


int main ()
{
	 //Mat Screen = imread("original.jpg",1);
	Mat Screen = imread("Ory.jpg",1);
	 Mat Src = imread("Rect.jpg",1);
	 Mat Dst = imread("RectD.jpg",1);
	 if(Screen.empty() || Src.empty() || Dst.empty() )
	 {
		cout<<"Image Loading failed"<<endl;
		getchar();
		return -1;
	 }
	 Mat Gray,src_gray;
	 //CS
	 cvtColor(Screen,Gray,CV_BGR2GRAY);
	 blur(Gray, Gray, Size(5,5));

	 Mat Border = Mat(Gray.size(),CV_8UC1);
	 adaptiveThreshold(Gray,Border,255,ADAPTIVE_THRESH_MEAN_C,THRESH_BINARY,5,2);
	 bitwise_not(Border,Border);
	 Mat Window=(Mat_<uchar>(3,3)<<0,1,0,1,1,1,0,1,0);
	 dilate(Border,Border,Window);

	 int count =0;
	 int max=-1;

	 Point MaxPt;

	 for ( int y=0; y< Border.size().height;y++)
	 {
		 uchar *row=Border.ptr(y);
		 for( int x=0;x<Border.size().width;x++)
		 {
			if (row[x]>128)
			{
			  int area=floodFill(Border,Point(x,y),CV_RGB(0,0,64));
			  if(area>max)
			  {
				MaxPt=Point(x,y);
				max=area;
			  }
			}
		 }
	 }
	 floodFill(Border, MaxPt, CV_RGB(255,255,255));
	 for(int y=0;y<Border.size().height;y++)
	 {
		  uchar * row=Border.ptr(y);
		  for(int x=0;x<Border.size().width;x++)
		  {
			  if(row[x]==64 && x!=MaxPt.x && y!=MaxPt.y)
			  {
				  int area=floodFill(Border,Point(x,y),CV_RGB(0,0,0));
			  }
		  }
	 }
 
	 erode(Border,Border,Window);
	 
	 vector <Point2f> features;
	 int maxLength;
	 vector<Vec2f> linesCS;
	 
	 //PC
	 vector<Vec2f> linesPC;
	 Mat HSV, Temp;
	 Mat DHSV, DTemp;
	 int iLowH = 170; int iHighH = 179;
	 int iLowS = 150; int iHighS = 255;
	 int iLowV = 60; int iHighV = 255;

	 /////////////////////////////////////////////////////

	 HoughLines(Border,linesCS,1,CV_PI/180,200);
	 mergeRelatedLinesCS(&linesCS,Gray);
	 Vec2f topEdgeCS=Vec2f(1000,1000);	Vec2f bottomEdgeCS=Vec2f(-1000,-1000);
	 Vec2f leftEdgeCS=Vec2f(1000,1000);	Vec2f rightEdgeCS=Vec2f(-1000,-1000);
	 
	 //Drawing CS1
	 drawlines(linesCS,topEdgeCS,bottomEdgeCS,rightEdgeCS,leftEdgeCS);
	 drawLine(topEdgeCS,Gray,CV_RGB(0,0,0));
	 drawLine(bottomEdgeCS,Gray,CV_RGB(0,0,0));
	 drawLine(leftEdgeCS,Gray,CV_RGB(0,0,0));
	 drawLine(rightEdgeCS,Gray,CV_RGB(0,0,0));
	 //imshow("EdgesGray", Gray);

	 CvPoint ptTopLeftCS, ptTopRightCS, ptBotLeftCS, ptBotRightCS;
	 intersectionPoints(Border,topEdgeCS,bottomEdgeCS,rightEdgeCS,leftEdgeCS,ptTopLeftCS,ptTopRightCS,ptBotLeftCS,ptBotRightCS);

	 //Drawing CS2
	 circle(Gray,ptTopLeftCS,10,Scalar(0,0,0),1);
	 circle(Gray,ptTopRightCS,10,Scalar(0,0,0),1);
	 circle(Gray,ptBotLeftCS,10,Scalar(0,0,0),1);
	 circle(Gray,ptBotRightCS,10,Scalar(0,0,0),1);
	 //imshow("Intersection Points Gray",Gray);

	 findMaxLength(maxLength,ptTopLeftCS,ptTopRightCS,ptBotLeftCS,ptBotRightCS);
	 // maxlength 304pix, other 285pix(304-19), plane actual=40.64x38.1
	 Point2f src[4], dst[4];
	 src[0]=ptTopLeftCS; dst[0]=Point2f(0,0);
	 src[1]=ptTopRightCS; dst[1]=Point2f(maxLength-1,0);
	 src[2]=ptBotRightCS; dst[2]=Point2f(maxLength-1,maxLength-20);
	 src[3]=ptBotLeftCS; dst[3]=Point2f(0,maxLength-20);

	 Mat C=getPerspectiveTransform(src,dst);

	 Mat Undistorted = Mat(Size(maxLength,maxLength-19),CV_8UC1);
	 cv::warpPerspective(Gray,Undistorted,cv::getPerspectiveTransform(src,dst),Size(maxLength,maxLength-19));
	 
	 //cout<<"The maximum Length in pixels is :"<<maxLength<<endl;

	 // Showing result
	 //imshow("Undistorted",Undistorted);

	 /////////////////////////////////////////////////////

	 /////////////////////////////////////////////////////

	 cvtColor(Src,HSV,COLOR_BGR2HSV);
	 cvtColor(Dst,DHSV,COLOR_BGR2HSV);
	
	 inRange(HSV,Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV),Temp);
	 inRange(DHSV,Scalar(iLowH, iLowS-130, iLowV), Scalar(iHighH, iHighS, iHighV),DTemp);
	
	 erode(Temp,Temp,getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
	 dilate(Temp,Temp,getStructuringElement(MORPH_ELLIPSE, Size(5, 5))); 
	 dilate(Temp,Temp,getStructuringElement(MORPH_ELLIPSE, Size(5, 5))); 
	 erode(Temp,Temp,getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
	
	 dilate(DTemp,DTemp,getStructuringElement(MORPH_ELLIPSE, Size(3, 3))); 
	 erode(DTemp,DTemp,getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
	
	 src_gray=Temp.clone();
	 goodFeaturesToTrack(src_gray,features,4,0.01,50,Mat(),3,false,0.04);
	 cornerSubPix(src_gray,features,Size(5,5),Size(-1,-1),cvTermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03 ) );

	 for (int i=0;i<4;i++)
	 {
		 circle(Src,features[i],10,Scalar(0,255,0),1,8);
	 }

	 //Showing result
	 //imshow("Src", Src);

	 HoughLines(DTemp,linesPC,1,CV_PI/180,200);
	 mergeRelatedLinesPC(&linesPC,DTemp);
	 
	 Vec2f topEdgePC=Vec2f(1000,1000);		Vec2f bottomEdgePC=Vec2f(-1000,-1000); 
	 Vec2f leftEdgePC=Vec2f(1000,1000);		Vec2f rightEdgePC=Vec2f(-1000,-1000); 
 
	 // Drawing PC1
	 drawlines(linesPC,topEdgePC,bottomEdgePC,rightEdgePC,leftEdgePC);
	 drawLine(topEdgePC,Dst,CV_RGB(0,255,0));
	 drawLine(bottomEdgePC,Dst,CV_RGB(0,255,0));
	 drawLine(leftEdgePC,Dst,CV_RGB(0,255,0));
	 drawLine(rightEdgePC,Dst,CV_RGB(0,255,0));
	 //imshow("EdgesDst", Dst);

	 CvPoint ptTopLeftPC, ptTopRightPC, ptBotLeftPC, ptBotRightPC;
	 intersectionPoints(Dst,topEdgePC,bottomEdgePC,rightEdgePC,leftEdgePC,ptTopLeftPC,ptTopRightPC,ptBotLeftPC,ptBotRightPC);

	 //Drawing RC2
	 circle(Dst,ptTopLeftPC,5,Scalar(0,255,0),1);
	 circle(Dst,ptTopRightPC,5,Scalar(0,255,0),1);
	 circle(Dst,ptBotLeftPC,5,Scalar(0,255,0),1);
	 circle(Dst,ptBotRightPC,5,Scalar(0,255,0),1);
	 //imshow("Intersection Points Dst",Dst);

	 vector <Point2f> srcpt, dstpt;
	 srcpt.push_back(features[1]); dstpt.push_back(ptTopLeftPC);
	 srcpt.push_back(features[2]); dstpt.push_back(ptTopRightPC);
	 srcpt.push_back(features[0]); dstpt.push_back(ptBotLeftPC);
	 srcpt.push_back(features[3]); dstpt.push_back(ptBotRightPC);

	 //Mat H= findHomography(srcpt,dstpt,CV_RANSAC);
	 Mat T = getPerspectiveTransform(srcpt,dstpt);
	 Mat circle=imread("circlegrid.jpg",1);
	 Mat OutBlah, OutReal;
 
	 warpPerspective(Src,OutBlah,T,Dst.size());
	 // Showing result
	 //imshow("OUTBLAH",OutBlah);

	 Mat S = (Mat_<double>(3,3) << 0.225, 0, 365, 0, 0.37, 205, 0, 0, 1);
	 Mat P=(C.inv())*T;
	 Mat W=P.inv()*S;
	 cout<<"C ="<<endl<<" "<<C<<endl<<endl;
	 cout<<"T ="<<endl<<" "<<T<<endl<<endl;
	 cout<<"P ="<<endl<<" "<<P<<endl<<endl;
	 cout<<"S ="<<endl<<" "<<S<<endl<<endl;
	 cout<<"W ="<<endl<<" "<<W<<endl<<endl;
	 warpPerspective(Src,OutReal,W,Src.size());
	 imwrite( "../OutReal.jpg", OutReal );
	 imshow("OUTREAL",OutReal);

	 waitKey(0);
	 return 0;
}
