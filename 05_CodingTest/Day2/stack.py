def solution(s):
    stack = []
    for c in s:
        if c == '(':
            stack.append(c)
        
        elif c == ')':
            if stack and stack[-1] == '(':
                stack.pop()
            else:
                return False
                
    return not stack