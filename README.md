# Aegis KUCIS TIL

```
KUCIS 팀의 TIL Repository 입니다.
```

### ⚙️ 세팅하기

1. 원격 저장소 클론하기

   ```bash
   git clone git@github.com:Aegis-2021/KUCIS-TIL.git
   ```

2. 패키지 설치하기

   ```bash
   yarn
   ```

3. 새로운 브랜치 생성하기

   ```bash
   git checkout -b til/{Github 닉네임}  # git checkout -b til/himitery
   ```

### ⚙️ TIL 작성하기

1. 오늘의 TIL을 생성합니다.

   ```bash
   yarn make {이름}  # ex) yarn make 이학진
   ```

2. TIL/이름 디렉토리로 이동합니다.

3. 이름(오늘의 날짜).md 파일에 TIL을 작성합니다.

4. 작성이 완료되었다면 커밋 후 원격 저장소에 푸시합니다.

   ```bash
   git commit -m "{이름}"  # git commit -m "이학진"
   git push origin til/{Github 닉네임}  # git push origin til/himitery
   ```

5. Github에서 PULL REQUEST를 작성합니다.

   ```
   주의) PULL REQUEST를 생성 후 merge 하지 않습니다.
   ```
