# Copyright (C) 2023 by Arunachala Amuda Murugan
# 
# Lisence: GNU General Public License v3.0

class Node:
    def __init__(self, data: int):
        """Node class for linked list

        Args:
            data (int): data to be stored in the node. (integer doc ID)
            next (Node, optional): next node in the linked list. Defaults to None.
            curr (Node, optional): current node in the linked list. Defaults to None.

        Returns:
            None
        """
        self.data: int = data
        self.next: Node | None = None
        self.curr = None


class LinkedList:
    def __init__(self):
        """Linked list class

        Args:
            head (Node, optional): head of the linked list. Defaults to None.
            curr (Node, optional): current node in the linked list. Defaults to None.

        Returns:
            None
        """
        self.head = None
        self.curr = None

    def __len__(self):
        """Length of the linked list

        Returns:
            int: length of the linked list
        """
        length: int = 0
        while self.curr:
            length += 1
            self.curr = self.curr.next
        return length

    def __str__(self):
        """String representation of the linked list

        Returns:
            str: string representation of the linked list
        """
        return " -> ".join(str(node.data) for node in self)

    def __next__(self):
        """Next node in the linked list

        Raises:
            StopIteration: if the linked list is empty or we have reached the end of the linked list

        Returns:
            Node: next node in the linked list
        """
        if self.curr is None:
            self.curr = self.head
        else:
            self.curr = self.curr.next
        if self.curr is None:
            raise StopIteration
        return self.curr

    def __iter__(self):
        """Iterator for the linked list

        Returns:
            LinkedList: itself
        """
        return self

    def is_present(self, data: int):
        """Check if the data is present in the linked list

        Args:
            data (int): data to be searched in the linked list

        Returns:
            bool: True if the data is present in the linked list, False otherwise
        """
        for node in self:
            if node.data == data:
                return True
        return False

    def sort(self):
        """Sort the linked list

        Returns:
            None
        """
        self.head = self._sort(self.head)

    def _sort(self, node: Node):
        """Helper recursive function to sort the linked list using merge sort

        Args:
            node (Node): node to be sorted

        Returns:
            Node: sorted node (head)
        """
        if node is None or node.next is None:
            return node
        left, right = self._split(node)
        left = self._sort(left)
        right = self._sort(right)
        return self._merge(left, right)

    def _split(self, node: Node):
        """Helper function to split the linked list into two halves

        Args:
            node (Node): node at which the linked list is to be split

        Returns:
            Node: head of the left half
            Node: head of the right half
        """
        if node is None or node.next is None:
            return node, None
        slow: Node = node
        fast: Node = node.next
        while fast:
            fast = fast.next
            if fast:
                fast = fast.next
                slow = slow.next
        mid = slow.next
        slow.next = None
        return node, mid

    def _merge(self, left: Node, right: Node):
        """Helper function to merge two sorted linked lists

        Args:
            left (Node): head of the left linked list
            right (Node): head of the right linked list

        Returns:
            Node: head of the merged linked list
        """
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

    def append(self, data: int):
        """Append a node to the end of the linked list

        Args:
            data (int): data to be stored in the node

        Returns:
            None
        """
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
            return
        last_node = self.head
        while last_node.next:
            last_node = last_node.next
        last_node.next = new_node

    def print_list(self):
        """Print the linked list

        Returns:
            None
        """
        curr_node = self.head
        while curr_node:
            print(curr_node.data, end=" -> ")
            curr_node = curr_node.next
        print("None\n")
