# Implementation OOO of Data-Parallel on BytePS

## docker file을 받아서 실행
## aws에 우리가 만들어둔 이미지를 사용해서 실행.



## 1. Setup
1. Launch 1 ~ 12 AWS p3.8xlarge instances
1. Copy the downloaded images to the AWS instances
1. Run the following commands in all the instances

## 2. Run on AWS
- Environment setting
  - Amazon AWS p3.8xlarge


```bash
$ docker build -f ./docker/Dockerfile .
```

```bash
$ ./scripts/remote/run.sh
```
