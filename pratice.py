user_input = '90 92 94 95'
hourly_temperature = user_input.split(' ')

for i in hourly_temperature:
    print(f'{i}', end=' ')
    if i == hourly_temperature[-1]:
        pass
    else:
        print(' -> ', end=' ')