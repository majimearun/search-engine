class BinaryTreeNode:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None
        self.is_word = False


class BinaryTree:
    def __init__(self):
        self.root = None

    def insert(self, data):
        self.root = self._insert(data, self.root)
        return self.root is not None

    def _insert(self, data, node):
        if node is None:
            node = BinaryTreeNode(data)
        else:
            if data <= node.data:
                node.left = self._insert(data, node.left)
            else:
                node.right = self._insert(data, node.right)
        return node

    def find(self, key) -> BinaryTreeNode:
        return self._find(key, self.root)

    def _find(self, key, node):
        if node is None or node.data == key:
            return node
        if key < node.data:
            return self._find(key, node.left)
        return self._find(key, node.right)

    def find_prefix(self, key) -> BinaryTreeNode:
        return self._find_prefix(key, self.root)

    def _find_prefix(self, key, node):
        if node is None:
            return None
        if node.data == key:
            return node
        if key < node.data:
            return self._find_prefix(key, node.left)
        return self._find_prefix(key, node.right)

    def find_prefixes(self, key) -> list:
        return self._find_prefixes(key, self.root)

    def _find_prefixes(self, key, node, prefixes=[]):
        if node is None:
            return prefixes
        if node.data == key:
            return prefixes
        if key < node.data:
            prefixes.append(node.data)
            return self._find_prefixes(key, node.left, prefixes)
        prefixes.append(node.data)
        return self._find_prefixes(key, node.right, prefixes)

    def find_all_words(self, node, words=[]):
        if node is None:
            return words
        if node.is_word:
            words.append(node.data)
        words = self.find_all_words(node.left, words)
        words = self.find_all_words(node.right, words)
        return words

    def find_all_words_with_prefix(self, key):
        node = self.find_prefix(key)
        return self.find_all_words(node)

    def find_all_words_with_prefixes(self, key):
        prefixes = self.find_prefixes(key)
        words = []
        for prefix in prefixes:
            words += self.find_all_words_with_prefix(prefix)
        return words

    def print_binary_tree(self):
        self._print_binary_tree(self.root)

    def _print_binary_tree(self, node):
        if node is None:
            return
        print(node.data)
        self._print_binary_tree(node.left)
        self._print_binary_tree(node.right)
