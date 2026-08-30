#pragma once
// Adapted from GMS-Feature-Matcher by JiaWang Bian.
// See ../LICENSE.GMS for the retained BSD-3-Clause notice.
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <ctime>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include "m2m_loransac/hopcroft_karp.h"
using namespace std;
using namespace cv;

const double DEFAULT_THRESH_FACTOR_M2M = 6.0;

// 8 possible rotation and each one is 3 X 3
const int mRotationPatterns_m2m[8][9] = {
	1,2,3,
	4,5,6,
	7,8,9,

	4,1,2,
	7,5,3,
	8,9,6,

	7,4,1,
	8,5,2,
	9,6,3,

	8,7,4,
	9,5,1,
	6,3,2,

	9,8,7,
	6,5,4,
	3,2,1,

	6,9,8,
	3,5,7,
	2,1,4,

	3,6,9,
	2,5,8,
	1,4,7,

	2,3,6,
	1,5,9,
	4,7,8
};

// 5 level scales
const double mScaleRatios_m2m[5] = { 1.0, 1.0 / 2, 1.0 / sqrt(2.0), sqrt(2.0), 2.0 };


class gms_matcher_m2m
{
public:
	// OpenCV Keypoints & Correspond Image Size & Nearest Neighbor Matches
	gms_matcher_m2m(const vector<KeyPoint> &vkp1, const Size size1, const vector<KeyPoint> &vkp2, const Size size2, const vector<DMatch> &vDMatches, const double thresh_factor = DEFAULT_THRESH_FACTOR_M2M, const int grid_size = 20)
	{
		// Input initialize
		NormalizePoints(vkp1, size1, mvP1);
		NormalizePoints(vkp2, size2, mvP2);
		mNumberMatches = vDMatches.size();
		mThreshFactor = thresh_factor;
		ConvertMatches(vDMatches, mvMatches);

		// Grid initialize
		mGridSizeLeft = Size(grid_size, grid_size);
		mGridNumberLeft = mGridSizeLeft.width * mGridSizeLeft.height;

		// Initialize the neihbor of left grid
		mGridNeighborLeft = Mat::zeros(mGridNumberLeft, 9, CV_32SC1);
		InitalizeNiehbors(mGridNeighborLeft, mGridSizeLeft);
	};
	~gms_matcher_m2m() {};

private:

	// Normalized Points
	vector<Point2f> mvP1, mvP2;

	// Matches (first = left compact idx, second = right compact idx)
	vector<pair<int, int> > mvMatches;

	// Number of Matches
	size_t mNumberMatches;
	double mThreshFactor;

	// Grid Size
	Size mGridSizeLeft, mGridSizeRight;
	int mGridNumberLeft;
	int mGridNumberRight;

	// x	  : left grid idx
	// y      : right grid idx
	// value  : max bipartite matching cardinality between cell pair
	Mat mMotionStatistics;

	// Number of unique left keypoints per left cell
	vector<int> mNumberPointsInPerCellLeft;

	// Inldex  : grid_idx_left
	// Value   : grid_idx_right
	vector<int> mCellPairs;

	// Every Match has a cell-pair
	// first  : grid_idx_left
	// second : grid_idx_right
	vector<pair<int, int> > mvMatchPairs;

	// Inlier Mask for output
	vector<bool> mvbInlierMask;

	//
	Mat mGridNeighborLeft;
	Mat mGridNeighborRight;

public:

	// Get Inlier Mask
	// Return number of inliers
	int GetInlierMask(vector<bool> &vbInliers, bool WithScale = false, bool WithRotation = false);

private:

	// Normalize Key Points to Range(0 - 1)
	void NormalizePoints(const vector<KeyPoint> &kp, const Size &size, vector<Point2f> &npts) {
		const size_t numP = kp.size();
		const int width   = size.width;
		const int height  = size.height;
		npts.resize(numP);

		for (size_t i = 0; i < numP; i++)
		{
			npts[i].x = kp[i].pt.x / width;
			npts[i].y = kp[i].pt.y / height;
		}
	}

	// Convert OpenCV DMatch to Match (pair<int, int>)
	void ConvertMatches(const vector<DMatch> &vDMatches, vector<pair<int, int> > &vMatches) {
		vMatches.resize(mNumberMatches);
		for (size_t i = 0; i < mNumberMatches; i++)
		{
			vMatches[i] = pair<int, int>(vDMatches[i].queryIdx, vDMatches[i].trainIdx);
		}
	}

	int GetGridIndexLeft(const Point2f &pt, int type) {
		int x = 0, y = 0;

		if (type == 1) {
			x = floor(pt.x * mGridSizeLeft.width);
			y = floor(pt.y * mGridSizeLeft.height);

			if (y >= mGridSizeLeft.height || x >= mGridSizeLeft.width){
				return -1;
			}
		}

		if (type == 2) {
			x = floor(pt.x * mGridSizeLeft.width + 0.5);
			y = floor(pt.y * mGridSizeLeft.height);

			if (x >= mGridSizeLeft.width || x < 1) {
				return -1;
			}
		}

		if (type == 3) {
			x = floor(pt.x * mGridSizeLeft.width);
			y = floor(pt.y * mGridSizeLeft.height + 0.5);

			if (y >= mGridSizeLeft.height || y < 1) {
				return -1;
			}
		}

		if (type == 4) {
			x = floor(pt.x * mGridSizeLeft.width + 0.5);
			y = floor(pt.y * mGridSizeLeft.height + 0.5);

			if (y >= mGridSizeLeft.height || y < 1 || x >= mGridSizeLeft.width || x < 1) {
				return -1;
			}
		}

		return x + y * mGridSizeLeft.width;
	}

	int GetGridIndexRight(const Point2f &pt) {
		int x = floor(pt.x * mGridSizeRight.width);
		int y = floor(pt.y * mGridSizeRight.height);

		return x + y * mGridSizeRight.width;
	}

	// Assign Matches to Cell Pairs (m2m-aware: counts unique left keypoints and
	// uses maximum bipartite matching cardinality via Hopcroft-Karp)
	void AssignMatchPairs(int GridType);

	// Verify Cell Pairs
	void VerifyCellPairs(int RotationType);

	// Get Neighbor 9
	vector<int> GetNB9(const int idx, const Size& GridSize) {
		vector<int> NB9(9, -1);

		int idx_x = idx % GridSize.width;
		int idx_y = idx / GridSize.width;

		for (int yi = -1; yi <= 1; yi++)
		{
			for (int xi = -1; xi <= 1; xi++)
			{
				int idx_xx = idx_x + xi;
				int idx_yy = idx_y + yi;

				if (idx_xx < 0 || idx_xx >= GridSize.width || idx_yy < 0 || idx_yy >= GridSize.height)
					continue;

				NB9[xi + 4 + yi * 3] = idx_xx + idx_yy * GridSize.width;
			}
		}
		return NB9;
	}

	void InitalizeNiehbors(Mat &neighbor, const Size& GridSize) {
		for (int i = 0; i < neighbor.rows; i++)
		{
			vector<int> NB9 = GetNB9(i, GridSize);
			int *data = neighbor.ptr<int>(i);
			memcpy(data, &NB9[0], sizeof(int) * 9);
		}
	}

	void SetScale(int Scale) {
		// Set Scale
		mGridSizeRight.width = mGridSizeLeft.width  * mScaleRatios_m2m[Scale];
		mGridSizeRight.height = mGridSizeLeft.height * mScaleRatios_m2m[Scale];
		mGridNumberRight = mGridSizeRight.width * mGridSizeRight.height;

		// Initialize the neihbor of right grid
		mGridNeighborRight = Mat::zeros(mGridNumberRight, 9, CV_32SC1);
		InitalizeNiehbors(mGridNeighborRight, mGridSizeRight);
	}

	// Run
	int run(int RotationType);
};

inline int gms_matcher_m2m::GetInlierMask(vector<bool> &vbInliers, bool WithScale, bool WithRotation) {

	int max_inlier = 0;

	if (!WithScale && !WithRotation)
	{
		SetScale(0);
		max_inlier = run(1);
		vbInliers = mvbInlierMask;
		return max_inlier;
	}

	// Scale/rotation search only updates vbInliers on improvement; ensure a
	// valid all-false mask when every run returns zero inliers.
	vbInliers.assign(mNumberMatches, false);

	if (WithRotation && WithScale)
	{
		for (int Scale = 0; Scale < 5; Scale++)
		{
			SetScale(Scale);
			for (int RotationType = 1; RotationType <= 8; RotationType++)
			{
				int num_inlier = run(RotationType);

				if (num_inlier > max_inlier)
				{
					vbInliers = mvbInlierMask;
					max_inlier = num_inlier;
				}
			}
		}
		return max_inlier;
	}

	if (WithRotation && !WithScale)
	{
		SetScale(0);
		for (int RotationType = 1; RotationType <= 8; RotationType++)
		{
			int num_inlier = run(RotationType);

			if (num_inlier > max_inlier)
			{
				vbInliers = mvbInlierMask;
				max_inlier = num_inlier;
			}
		}
		return max_inlier;
	}

	if (!WithRotation && WithScale)
	{
		for (int Scale = 0; Scale < 5; Scale++)
		{
			SetScale(Scale);

			int num_inlier = run(1);

			if (num_inlier > max_inlier)
			{
				vbInliers = mvbInlierMask;
				max_inlier = num_inlier;
			}

		}
		return max_inlier;
	}

	return max_inlier;
}

inline void gms_matcher_m2m::AssignMatchPairs(int GridType) {
	// Pass 1: bucket associations by cell pair and collect unique left IDs per cell.
	//
	// unique_left[lgidx]  = set of distinct left compact indices in that cell
	// cell_edges[key]     = set of (left_compact, right_compact) edges for the cell
	//                       pair encoded as key = lgidx * mGridNumberRight + rgidx.
	//                       Using std::set deduplicates identical (l, r) rows.
	unordered_map<int, unordered_set<int>>         unique_left;
	unordered_map<int, set<pair<int,int>>>          cell_edges;

	for (size_t i = 0; i < mNumberMatches; i++)
	{
		Point2f &lp = mvP1[mvMatches[i].first];
		Point2f &rp = mvP2[mvMatches[i].second];

		int lgidx = mvMatchPairs[i].first = GetGridIndexLeft(lp, GridType);
		int rgidx = -1;

		if (GridType == 1)
		{
			rgidx = mvMatchPairs[i].second = GetGridIndexRight(rp);
		}
		else
		{
			rgidx = mvMatchPairs[i].second;
		}

		if (lgidx < 0 || rgidx < 0) continue;

		unique_left[lgidx].insert(mvMatches[i].first);
		cell_edges[lgidx * mGridNumberRight + rgidx].insert({mvMatches[i].first, mvMatches[i].second});
	}

	// Pass 2a: fill mNumberPointsInPerCellLeft with unique left keypoint counts.
	for (const auto &kv : unique_left)
		mNumberPointsInPerCellLeft[kv.first] = static_cast<int>(kv.second.size());

	// Pass 2b: for each cell pair compute the maximum bipartite matching
	// cardinality via Hopcroft-Karp and store it in mMotionStatistics.
	for (const auto &kv : cell_edges)
	{
		const int key   = kv.first;
		const auto &edges = kv.second;
		const int lgidx = key / mGridNumberRight;
		const int rgidx = key % mGridNumberRight;

		// Remap global compact IDs to zero-based local indices. Iterating the
		// ordered edge set makes both the graph and its traversal deterministic.
		unordered_map<int,int> lmap, rmap;
		for (const auto &e : edges)
		{
			lmap.emplace(e.first,  static_cast<int>(lmap.size()));
			rmap.emplace(e.second, static_cast<int>(rmap.size()));
		}

		dino_m2m::HopcroftKarp graph(lmap.size(), rmap.size());
		for (const auto &e : edges)
			graph.add_edge(static_cast<size_t>(lmap.at(e.first)), static_cast<size_t>(rmap.at(e.second)));

		mMotionStatistics.at<int>(lgidx, rgidx) = static_cast<int>(graph.maximum_matching());
	}
}

inline void gms_matcher_m2m::VerifyCellPairs(int RotationType) {

	const int *CurrentRP = mRotationPatterns_m2m[RotationType - 1];

	for (int i = 0; i < mGridNumberLeft; i++)
	{
		if (sum(mMotionStatistics.row(i))[0] == 0)
		{
			mCellPairs[i] = -1;
			continue;
		}
		// For each left patch, find the right patch with the most matches
		int max_number = 0;
		for (int j = 0; j < mGridNumberRight; j++)
		{
			int *value = mMotionStatistics.ptr<int>(i);
			if (value[j] > max_number)
			{
				mCellPairs[i] = j;
				max_number = value[j];
			}
		}

		int idx_grid_rt = mCellPairs[i];

		const int *NB9_lt = mGridNeighborLeft.ptr<int>(i);
		const int *NB9_rt = mGridNeighborRight.ptr<int>(idx_grid_rt);

		int score = 0;
		int point_number = 0;
		double thresh = 0;
		int numpair = 0;

		for (size_t j = 0; j < 9; j++)
		{
			int ll = NB9_lt[j];
			int rr = NB9_rt[CurrentRP[j] - 1];
			if (ll == -1 || rr == -1)	continue;
			score += mMotionStatistics.at<int>(ll, rr);
			point_number += mNumberPointsInPerCellLeft[ll];
			numpair++;
		}

		thresh = mThreshFactor * sqrt(point_number / numpair);

		if (score < thresh)
			mCellPairs[i] = -2;
	}
}

inline int gms_matcher_m2m::run(int RotationType) {

	mvbInlierMask.assign(mNumberMatches, false);

	// Initialize Motion Statistics
	mMotionStatistics = Mat::zeros(mGridNumberLeft, mGridNumberRight, CV_32SC1);
	mvMatchPairs.assign(mNumberMatches, pair<int, int>(0, 0));

	for (int GridType = 1; GridType <= 4; GridType++)
	{
		// initialize
		mMotionStatistics.setTo(0);
		mCellPairs.assign(mGridNumberLeft, -1);
		mNumberPointsInPerCellLeft.assign(mGridNumberLeft, 0);

		AssignMatchPairs(GridType);
		VerifyCellPairs(RotationType);

		// Mark inliers
		for (size_t i = 0; i < mNumberMatches; i++)
		{
			if (mvMatchPairs[i].first >= 0) {
				if (mCellPairs[mvMatchPairs[i].first] == mvMatchPairs[i].second)
				{
					mvbInlierMask[i] = true;
				}
			}
		}
	}
	int num_inlier = sum(mvbInlierMask)[0];
	return num_inlier;
}
