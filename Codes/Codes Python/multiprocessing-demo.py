import concurrent.futures
import time

start = time.perf_counter()

result_list = []


def do_something(seconds):
	print(f'Sleeping {seconds} second(s)...')
	time.sleep(seconds)
	return f'Done Sleeping...{seconds}'


l_function = lambda x: x ** 2

if __name__ == '__main__':
	with concurrent.futures.ProcessPoolExecutor() as executor:
		secs = [5, 4, 3, 2, 1]
		results = executor.map(do_something, secs)

		for result in results:
			result_list.append(result)

	finish = time.perf_counter()

	print(f'Finished in {round(finish - start, 2)} second(s)')
