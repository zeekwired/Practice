user_input = '1 2 3, 4 5 6, 7 8 9, 10 11 12, 13 14 15'
lines = user_input.split(',')

mult_table = [[int(num) for num in line.split()] for line in lines]

for row in mult_table:
    for num in row:
        if num < max(row):
            print(num, end=' | ')
        else:
            print(num, end='')
    print()