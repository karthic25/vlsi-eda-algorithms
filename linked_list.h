template <typename T>
class LinkedList {
	int m_nElems = 0;
	vector<T> m_items {};

	public:

    LinkedList(int n, vector<T> items) {
        m_nElems = n;
        m_items = items;
    }

    void push_item(T item) {
        m_nElems += 1;
        m_items.push_back(item);
    }
};