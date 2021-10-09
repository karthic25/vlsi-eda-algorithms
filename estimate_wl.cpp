#include <utility>
#include <iostream>
#include <vector>
#include "linked_list.h"

using namespace std;

LinkedList<pair<int, int>> shortest_path_bet_two_nodes(pair<int, int> node1, pair<int, int> node2, int nRows, int nCols) {
	LinkedList<pair<int, int>> ll(0, {});
	vector<vector<int>> matrix(nRows, vector<int>(nCols, 0));
	matrix[node1.first][node1.second] = 1;
	matrix[node2.first][node2.second] = 1;
	
	vector<vector<int>> scores(nRows, vector<int>(nCols, 0));

	for(int i=node1.first; i<nRows; i++) {
		for(int j=node1.first)
	}
}

int main() {
	cout << "Hello World!" << endl;
	return 0;
}