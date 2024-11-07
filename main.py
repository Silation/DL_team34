from concorde.tsp import TSPSolver


def read_tsp_file(file_path):
    # coordinates에 (x, y) tuple 저장
    coordinates = []
    start_reading = False
    
    with open(file_path, 'r') as file:
        for line in file:
            
            if line.strip() == "EOF":
                break
            
            if start_reading:
                _, x, y = line.split()
                coordinates.append((int(x), int(y)))

            if line.strip() == "NODE_COORD_SECTION":
                start_reading = True

    return coordinates

# 폴더 경로
folder_path = 'DL_TestCase/'
coordinates = read_tsp_file(folder_path + 'kroA200.tsp')

print(coordinates)

# TSPSolver를 사용하여 문제 인스턴스를 생성합니다.
solver = TSPSolver.from_data([x for x, y in coordinates], [y for x, y in coordinates], norm="EUC_2D")

# 최적화를 실행합니다.
solution = solver.solve()

# 결과 출력
print("Found tour with total length:", solution.optimal_value)
print("Tour order:", solution.tour)
