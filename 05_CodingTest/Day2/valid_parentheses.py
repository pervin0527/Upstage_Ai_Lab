class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        for c in s:
            if c in ['(', '{', '[']:
                stack.append(c)
            elif c == ')':
                if stack and stack[-1] == '(':
                    stack.pop()
                else:
                    return False
            elif c == '}':
                if stack and stack[-1] == '{':
                    stack.pop()
                else:
                    return False
            elif c == ']':
                if stack and stack[-1] == '[':
                    stack.pop()
                else:
                    return False

        return not stack
    
    def isValid2(self, s: str) -> bool:
        stack = []
        mapping = {')': '(', '}': '{', ']': '['}
        
        for c in s:
            if c in mapping.values():
                stack.append(c)
            elif c in mapping:
                if stack and stack[-1] == mapping[c]:
                    stack.pop()
                else:
                    return False

        return not stack