// Set new default font family and font color to mimic Bootstrap's default styling
Chart.defaults.global.defaultFontFamily = '-apple-system,system-ui,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif';
Chart.defaults.global.defaultFontColor = '#292b2c';

// 각 동별 평균 데시벨 차트
function fetchDongData() {
  // API 요청 및 데이터 가져오기 함수
  function fetchData(page) {
      const url = `https://management.noiroze.com/api/sound_level/?page=${page}`; // 페이지 번호에 따라 URL 동적 생성
      return fetch(url)
          .then(response => response.json())
          .then(data => data.results);
  }

  // // 모든 페이지의 데이터를 가져오는 함수
  // async function fetchAllData() {
  //     const totalPages = 15; // 전체 페이지 수
  //     const promises = [];

  //     for (let page = 1; page <= totalPages; page++) {
  //         promises.push(fetchData(page));
  //     }

  //     const results = await Promise.all(promises);
  //     // 결과 배열 합치기
  //     const mergedResults = results.flat();
  //     return mergedResults;
  // }

  // // 차트 데이터 업데이트 함수
  // function updateChartData() {
  //     fetchAllData()
  //         .then(data => {
  //             // 동별로 데이터 그룹화 및 평균 계산
  //             var groupedData = {};
  //             data.forEach(item => {
  //                 const dong = item.dong;
  //                 if (groupedData.hasOwnProperty(dong)) {
  //                     groupedData[dong].push(item.value);
  //                 } else {
  //                     groupedData[dong] = [item.value];
  //                 }
  //             });

  //             // 동별 평균 데이터 추출
  //             var labels = [];
  //             var values = [];
  //             for (const [dong, items] of Object.entries(groupedData)) {
  //                 labels.push(dong);
  //                 if (items.length === 0) {
  //                     values.push('');
  //                 } else {
  //                     var sum = items.reduce((total, value) => total + value, 0);
  //                     var average = sum / items.length;
  //                     values.push(average.toFixed(2));
  //                 }
  //             }

  //             // 차트 데이터 업데이트
  //             myAreaChart.data.labels = labels;
  //             myAreaChart.data.datasets[0].data = values;
  //             myAreaChart.update();
  //         })
  //         .catch(error => {
  //             console.error('Error:', error);
  //         });
  // }

  // 차트 생성 및 데이터 업데이트
  var ctx = document.getElementById('myAreaChart').getContext('2d');
  var myAreaChart = new Chart(ctx, {
      type: 'bar',
      data: {
          labels: [],
          datasets: [{
              label: 'Sound Level',
              data: [],
              backgroundColor: 'rgba(103, 104, 172, 0.8)',
              borderColor: 'rgba(103, 104, 172, 1)',
              borderWidth: 1
          }]
      },
      options: {
          scales: {
              xAxes: [{
                  ticks: {
                      autoSkip: false, // 각 동을 모두 표시하기 위해 autoSkip을 false로 설정
                  }
              }],
              yAxes: [{
                  ticks: {
                      beginAtZero: true
                  }
              }]
          }
      }
  });

  // 초기 데이터 업데이트
  updateChartData();
  // 매일 자정마다 데이터를 업데이트합니다.
  setInterval(updateChartData, 86400000); // 24시간(1일) 간격으로 업데이트
}


///////////////////////////
// 각 동별 데이터 갯수의 비율을 파이 차트로 나타내는 함수
function fetchTimeData() {
  // API 요청 및 데이터 가져오기 함수
  function fetchData(page) {
    const url = `https://management.noiroze.com/api/sound_level/?page=${page}`; // 페이지 번호에 따라 URL 동적 생성
    return fetch(url)
      .then(response => response.json())
      .then(data => data.results);
  }

  // 모든 페이지의 데이터를 가져오는 함수
  async function fetchAllData() {
    const totalPages = 15; // 전체 페이지 수
    const promises = [];

    for (let page = 1; page <= totalPages; page++) {
      promises.push(fetchData(page));
    }

    const results = await Promise.all(promises);
    // 결과 배열 합치기
    const mergedResults = results.flat();
    return mergedResults;
  }

  // 차트 데이터 업데이트 함수
  function updateChartData() {
    fetchAllData()
      .then(data => {
        // 동별로 데이터 그룹화
        var groupedData = {};
        data.forEach(item => {
          const dong = item.dong;
          if (groupedData.hasOwnProperty(dong)) {
            groupedData[dong].push(item);
          } else {
            groupedData[dong] = [item];
          }
        });

        // 각 동의 데이터 갯수 계산
        var labels = [];
        var values = [];
        for (const [dong, items] of Object.entries(groupedData)) {
          labels.push(dong);
          values.push(items.length.toFixed(1)); // 소수점 1자리까지 반올림하여 추가
        }

        // 차트 데이터 업데이트
        myBarChart.data.labels = labels;
        myBarChart.data.datasets[0].data = values;
        myBarChart.update();
      })
      .catch(error => {
        console.error('Error:', error);
      });
  }

  // 차트 생성 및 데이터 업데이트
  var ctx2 = document.getElementById('myBarChart').getContext('2d');
  var myBarChart = new Chart(ctx2, {
    type: 'pie',
    data: {
      labels: [],
      datasets: [{
        label: 'Data Count',
        data: [],
        backgroundColor: [
          '#2f3055',
          '#464780',
          '#6667ab',
          '#898abe',
          '#afb0d3',
          // 다른 동에 대한 색상을 필요에 따라 추가할 수 있습니다.
        ],
        borderColor: 'rgba(255, 255, 255, 1)',
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: true
    }
  });

  // 초기 데이터 업데이트
  updateChartData();
  // 매일 자정마다 데이터를 업데이트합니다.
  setInterval(updateChartData, 86400000); // 24시간(1일) 간격으로 업데이트
}

// fetchDongData() 함수 호출
fetchDongData();
fetchTimeData();


