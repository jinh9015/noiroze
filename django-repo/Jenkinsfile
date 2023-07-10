pipeline {
    agent any

    stages {
        stage("Checkout") {
            steps {
                checkout scm
            }
        }

        stage("Build") {
            steps {
                script {
                    // Docker 빌드 실행
                    sh "docker build -t jinh9015/noiroze-web ./NoiroseServer"
                }
            }
        }

        stage("Tag and Push") {
            steps {
                withCredentials([usernamePassword(
                    credentialsId: 'DockerHub',
                    usernameVariable: 'DOCKER_USER_ID',
                    passwordVariable: 'DOCKER_USER_PASSWORD'
                )]) {
                    script {
                        env.DOCKER_USER_ID = DOCKER_USER_ID
                        env.DOCKER_USER_PASSWORD = DOCKER_USER_PASSWORD
                        sh 'docker tag jinh9015/noiroze-web:latest $DOCKER_USER_ID/noiroze-web:$BUILD_NUMBER'
                        sh 'docker login -u $DOCKER_USER_ID -p $DOCKER_USER_PASSWORD'
                        sh 'docker push $DOCKER_USER_ID/noiroze-web:$BUILD_NUMBER'
                    }
                }
            }
        }
    }

    post {
        always {
            script {
                // 이미 클론된 레파지토리 확인 후 삭제
                def repoPath = "${env.WORKSPACE}/noiroze-web-manifest"
                if (fileExists(repoPath)) {
                    sh "rm -rf ${repoPath}"
                }

                // GitHub 토큰 가져오기
                withCredentials([usernamePassword(
                    credentialsId: 'github_token',
                    usernameVariable: 'GITHUB_USERNAME',
                    passwordVariable: 'GITHUB_TOKEN'
                )]) {
                    // Git 레파지토리 클론
                    sh "git clone -b main https://${GITHUB_USERNAME}:${GITHUB_TOKEN}@github.com/jinh9015/noiroze-web-manifest.git ${repoPath}"

                    // deployment.yaml 템플릿 파일 로드
                    def deploymentTemplatePath = "${env.WORKSPACE}/noiroze-web-manifest/deployment.yaml.template"
                    def deploymentTemplate = readFile(deploymentTemplatePath)

                    // 이미지 태그 동적으로 적용
                    def updatedDeploymentYaml = deploymentTemplate.replaceAll('\\$DOCKER_USER_ID', DOCKER_USER_ID)
                                                                 .replaceAll('\\$BUILD_NUMBER', BUILD_NUMBER)

                    // 동적으로 생성한 deployment.yaml 파일 저장
                    def deploymentYamlPath = "${env.WORKSPACE}/noiroze-web-manifest/deployment.yaml"
                    writeFile(file: deploymentYamlPath, text: updatedDeploymentYaml)

                    // Git 사용자 이름과 이메일 설정
                    sh "git config user.name '${GITHUB_USERNAME}'"
                    sh "git config user.email '${GITHUB_USERNAME}@gmail.com'"
                    sh "git config --global user.name '${GITHUB_USERNAME}'"
                    sh "git config --global user.email '${GITHUB_USERNAME}@gmail.com'"

                    // noiroze-web-manifest 레파지토리에 변경 사항을 커밋하고 푸시
                    dir("${env.WORKSPACE}/noiroze-web-manifest") {
                        sh 'git add deployment.yaml'
                        sh 'git commit -m "Update deployment.yaml - ${BUILD_NUMBER}"'
                        sh 'git push origin main'
                    }
                }
            }
        }
    }
}



