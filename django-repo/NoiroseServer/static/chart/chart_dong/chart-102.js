// Set new default font family and font color to mimic Bootstrap's default styling
Chart.defaults.global.defaultFontFamily = '-apple-system,system-ui,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif';
Chart.defaults.global.defaultFontColor = '#292b2c';


/////////////월별 평균 데시벨//////////////

function fetchMonthlyData() {
    // API 요청 및 데이터 가져오기 함수
    function fetchData(page) {
      const url = `https://management.noiroze.com/api/sound_level/?page=${page}`; // 페이지 번호에 따라 URL 동적 생성
      return fetch(url)
        .then(response => response.json())
        .then(data => data.results);
    }
  
    // 모든 페이지의 데이터를 가져오는 함수
    async function fetchAllData() {
      const totalPages = 70; // 전체 페이지 수
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
            // “dong” 값이 102 데이터 필터링
            var filteredData = data.filter(item => item.dong === '102');
            // 월별로 데이터 그룹화 및 평균 계산
            var groupedData = {};
            filteredData.forEach(item => {
            const createdAt = new Date(item.created_at);
            const year = createdAt.getFullYear();
            const month = createdAt.getMonth() + 1;
            const key = `${year}-${month}`;
            if (groupedData.hasOwnProperty(key)) {
                groupedData[key].push(item.value);
            } else {
                groupedData[key] = [item.value];
            }
            });
    
            // 최근 7개월의 데이터 추출
            var currentDate = new Date();
            var labels = [];
            var values = [];
            for (let i = 0; i < 7; i++) {
            var year = currentDate.getFullYear();
            var month = currentDate.getMonth() + 1;
            var key = `${year}-${month}`;
            labels.unshift(key);
            var items = groupedData[key] || [];
            if (items.length === 0) {
                values.unshift('');
            } else {
                var sum = items.reduce((total, value) => total + value, 0);
                var average = sum / items.length;
                values.unshift(average.toFixed(2));
            }
            currentDate.setMonth(currentDate.getMonth() - 1);
            }
            
            // 차트 데이터 업데이트
            myAreaChart.data.labels = labels;
            myAreaChart.data.datasets[0].data = values;
            myAreaChart.update();
        })
        .catch(error => {
            console.error('Error:', error);
        });
    }
  
    // 차트 생성 및 데이터 업데이트
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
              autoSkip: true,
              maxTicksLimit: 10,
              maxRotation: 0,
              callback: function(value, index) {
                if (value === '') return '';
                var [year, month] = value.split('-');
                return `${year}/${month}`;
              }
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
  
  //////////////////////////////////////////////////////// 일별 평균 데시벨 차트 ////////////////////////////////////////////////////////////////////////
  function fetchDailyData() {
    // API 요청 및 데이터 가져오기 함수
    function fetchData(page) {
      const url = `https://management.noiroze.com/api/sound_level/?page=${page}`; // 페이지 번호에 따라 URL 동적 생성
      return fetch(url)
        .then(response => response.json())
        .then(data => data.results);
    }
  
    // 모든 페이지의 데이터를 가져오는 함수
    async function fetchAllData() {
      const totalPages = 70; // 전체 페이지 수
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
          // “dong” 값이 102인 데이터 필터링
          var filteredData = data.filter(item => item.dong === '102');
          // 일별로 데이터 그룹화
          var groupedData = {};
          filteredData.forEach(item => {
            const createdAt = new Date(item.created_at);
            const date = createdAt.toISOString().split('T')[0];
            if (groupedData.hasOwnProperty(date)) {
              groupedData[date].push(item);
            } else {
              groupedData[date] = [item];
            }
          });
          // 최근 7일의 데이터 추출
          var currentDate = new Date();
          var labels = [];
          var values = [];
          for (let i = 0; i < 7; i++) {
            var date = currentDate.toISOString().split('T')[0];
            labels.unshift(date);
            var items = groupedData[date] || [];
            if (items.length === 0) {
              values.unshift('');
            } else {
              var sum = items.reduce((total, item) => total + item.value, 0);
              var average = sum / items.length;
              values.unshift(average.toFixed(2));
            }
            currentDate.setDate(currentDate.getDate() - 1);
          }
          // 차트 데이터 업데이트
          myBarChart.data.labels = labels;
          myBarChart.data.datasets[0].data = values;
          myBarChart.update();
        });
    }
  
    // 차트 생성 및 데이터 업데이트
    var myBarChart = new Chart(ctx2, {
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
              autoSkip: true,
              maxTicksLimit: 10,
              maxRotation: 0,
              callback: function(value, index) {
                if (value === '') return '';
                var date = new Date(value);
                var month = date.getMonth() + 1;
                var day = date.getDate();
                return month + '/' + day;
              }
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
  

    /////////////////////////////////////////////////////////////////////// 호수별 평균 데시벨 차트 /////////////////////////////////////////////////////////////
    function fetchHoData() {
        // 호수 리스트
        var hoList = ['101', '102', '201', '202', '301', '302', '401', '402', '501', '502', '601', '602', '701', '702', '801', '802', '901', '902', '1001', '1002'];
        // // 옵션 및 스타일 설정
    Chart.plugins.register({
        beforeDraw: function(chartInstance, easing) {
            if (chartInstance.chart.canvas.id === 'myHoChart') {
                var yScale = chartInstance.scales['y-axis-0'];
                var ctx = chartInstance.chart.ctx;

                // y=39 선 그리기
                var index39 = yScale.getPixelForValue(39);
                ctx.save();
                ctx.strokeStyle = '#f89b00'; // 보라색으로 설정
                ctx.lineWidth = 3; // 선 너비 3
                ctx.beginPath();
                ctx.moveTo(chartInstance.chartArea.left, index39);
                ctx.lineTo(chartInstance.chartArea.right, index39);
                ctx.stroke();
                ctx.restore();

                // y=34 선 그리기
                var index34 = yScale.getPixelForValue(34);
                ctx.save();
                ctx.strokeStyle = '#000e38'; // 주황색으로 설정
                ctx.lineWidth = 3; // 선 너비 3
                ctx.beginPath();
                ctx.moveTo(chartInstance.chartArea.left, index34);
                ctx.lineTo(chartInstance.chartArea.right, index34);
                ctx.stroke();
                ctx.restore();
            }
        }
    });
        var options = {
            scales: {
                yAxes: [{
                    ticks: {
                        beginAtZero: true
                    }
                }]
            },
            plugins: {
                // 플러그인 등록한 내용 추가
                beforeDraw: function(chartInstance, easing) {
                    // ...
                }
            }
        };
        // API 요청 및 데이터 가져오기 함수
        function fetchData(page) {
        const url = `https://management.noiroze.com/api/sound_level/?page=${page}`; // 페이지 번호에 따라 URL 동적 생성
        return fetch(url)
            .then(response => response.json())
            .then(data => data.results.filter(item => item.dong === '102')); // 102동 필터링 추가
        }
    
        // 모든 페이지의 데이터를 가져오는 함수
        async function fetchAllData() {
        const totalPages = 70; // 전체 페이지 수
        const promises = [];
    
        for (let page = 1; page <= totalPages; page++) {
            promises.push(fetchData(page));
        }
    
        const results = await Promise.all(promises);
        // 결과 배열 합치기
        const mergedResults = results.flat();
        return mergedResults.filter(item => hoList.includes(item.ho)); // 호수 필터링 추가
        }
    
        // 차트 데이터 업데이트 함수
        function updateChartData() {
        fetchAllData()
            .then(data => {
            // 호수별로 데이터 그룹화
            var groupedData = {};
            hoList.forEach(ho => {
                var filteredData = data.filter(item => item.ho === ho);
                groupedData[ho] = filteredData;
            });
    
            // 호수별 평균 데시벨 추출
            var labels = [];
            var values = [];
            hoList.forEach(ho => {
                var items = groupedData[ho] || [];
                var sum = items.reduce((total, item) => total + item.value, 0);
                var average = sum / items.length;
                labels.push(ho);
                values.push(average.toFixed(2));
            });
    
            // 차트 데이터 업데이트
            myHoChart.data.labels = labels;
            myHoChart.data.datasets[0].data = values;
            myHoChart.update();
            });
        }
    
        // 차트 생성 및 데이터 업데이트
        var myHoChart = new Chart(ctx3, {
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

    // 차트 초기화 및 데이터 업데이트
    var ctx = document.getElementById('myAreaChart').getContext('2d');
    var ctx2 = document.getElementById('myBarChart').getContext('2d');
    var ctx3 = document.getElementById('myHoChart').getContext('2d');
    fetchMonthlyData();
    fetchDailyData();
    fetchHoData();



