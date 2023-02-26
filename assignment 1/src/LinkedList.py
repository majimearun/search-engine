class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
        self.curr = None


class LinkedList:
    def __init__(self):
        self.head = None
        self.curr = None

    def __len__(self):
        length = 0
        while self.curr:
            length += 1
            self.curr = self.curr.next
        return length

    def __str__(self):
        return " -> ".join(str(node.data) for node in self)

    def __next__(self):
        if self.curr is None:
            self.curr = self.head
        else:
            self.curr = self.curr.next
        if self.curr is None:
            raise StopIteration
        return self.curr

    def __iter__(self):
        return self

    def is_present(self, data):
        for node in self:
            if node.data == data:
                return True
        return False

    def sort(self):
        self.head = self._sort(self.head)

    def _sort(self, node):
        if node is None or node.next is None:
            return node
        left, right = self._split(node)
        left = self._sort(left)
        right = self._sort(right)
        return self._merge(left, right)

    def _split(self, node):
        if node is None or node.next is None:
            return node, None
        slow = node
        fast = node.next
        while fast:
            fast = fast.next
            if fast:
                fast = fast.next
                slow = slow.next
        mid = slow.next
        slow.next = None
        return node, mid

    def _merge(self, left, right):
        if left is None:
            return right
        if right is None:
            return left
        if left.data <= right.data:
            result = left
            result.next = self._merge(left.next, right)
        else:
            result = right
            result.next = self._merge(left, right.next)
        return result

    def append(self, data):
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
            return
        last_node = self.head
        while last_node.next:
            last_node = last_node.next
        last_node.next = new_node

    def print_list(self):
        curr_node = self.head
        while curr_node:
            print(curr_node.data, end=" -> ")
            curr_node = curr_node.next
        print("None\n")
