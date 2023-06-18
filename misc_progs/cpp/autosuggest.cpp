#include <iostream>
#include <bits/stdc++.h>

using namespace std;

class Node
{
public:
	bool isWord;
	unordered_map<char, Node*> children;
	Node() {isWord = false;}
};

class Trie
{
private:
	Node *root;
	void postOrderClear(Node *cur)
	{
		if(cur == nullptr) return;
		if(cur->children.size() == 0) {
			delete cur;
			cur = nullptr;
			return;
		}
		for(auto ch : cur->children) {
			postOrderClear(ch.second);
		}
		return;
	}
public:
	Trie()
	{
		root = new Node();
	}
	~Trie()
	{
		postOrderClear(root);
	}
	void insert(string word)
	{
		if(root == nullptr) root = new Node();
		Node * cur = root;
		unordered_map<char, Node*>::const_iterator it;
		int n = word.length();
		int l = 0;
		for(auto c : word)
		{
			l++;
			it = cur->children.find(c);
		        if(it == cur->children.end())
			{
				cur->children[c] = new Node();
				if(l == n) cur->children[c]->isWord = true;
			}
		        cur = cur->children[c];
		}
	}
};

vector< vector<string> > threeKeywordSuggestions(int numreviews, vector<string> repository, string customerQuery)
{
	vector< vector<string> > ret;
	//Trie* t = new Trie();
	Trie t;
	for(auto word : repository)
	{
		t.insert(word);
	}
	return ret;
}

int main()
{
    vector<string> repo = {"mobile", "mouse", "moneypot", "monitor", "mousepad"};
    int nr = repo.size();
    string cq = "mouse";
    vector< vector<string> > ans = threeKeywordSuggestions(nr, repo, cq);
    cout << "printing output\n";
    int n = 1;
    for(auto a : ans)
    {
	   n++;
	   cout << cq.substr(0, n) << " : [";
	   for(auto w : a) cout <<  w << " ";
	   cout << "]\n";
    }
    return 0;
}
