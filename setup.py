from setuptools import setup, find_packages

setup_requires = [
    ]

install_requires = [
    'beautifulsoup4 == 4.10.0'
    ,'bs4 == 0.0.1'
    ,'fastapi == 0.75.0'
    ,'requests == 2.27.1'
    ,'urllib3 == 1.26.8'
    ,'uvicorn == 0.17.6'
    ,'matplotlib'
    ,'seaborn'
    ,'numpy'
    ,'pandas'
    ,'sklearn'
    ]

setup(
    #패키지 이름
    name='Flowdas-Books',
    #패키지 버전 이름
    version='0.1',
    #패키지 요약정보
    description='Flowdas Books',
    #패키지 작성자 정보
    author='Flowdas',
    #패키지 작성자 정보
    author_email='spammustdie@flowdas.com',
    #배포 패키지에 포함되어야 하는 python패키지, 수동으로 입력할 수도 있고 find_packages() 함수로 자동으로 모든 패키지와 서프 패키지 찾을수 있음
    packages=find_packages(),
    #라이브러리 등 명시
    install_requires=install_requires,
    #setup.py 자기 자신을 위한 필요 패키지 명시
    setup_requires=setup_requires,
    )