def print_progress(msg, i, size):
    end = '\n' if i+1 == size else '\r'
    print(f'{msg} {i+1}/{size}', end=end)
