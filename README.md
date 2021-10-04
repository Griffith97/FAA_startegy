FAA_startegy
=============
사용법
------
FAA(level, period, w1, w2, w3, number, transaction_cost)  
* level : 자산군(risky, safe)의 위험도 선택
   >Risky assets : XLY, XLV, XLU, XLP, XLK, XLI, XLF, XLE, XLB, VOX, RWR     
   >Safe assets : FXY, FXF, GLD, IEF, SH, TLT, SHY, SHV     
   >Cash : SHV

* period : 자산군의 모멘텀 또는 Permutation pattern, 변동성, 상관성 계산 주기(월 단위)
   > #### 모멘텀 종류   
   >   
   > 1.  상대 모멘텀   
   > 2.  Permutation pattern   
   > 3.  Kaufman efficiency ratio   
* w1 : 모멘텀 비중 또는 Permutation pattern 비중
* w2 : 변동성 비중
* w3 : 상관성 비중
* number : 자산군에 포함되는 종목 수
* transaction_cost : 거래비용 및 슬리피지
    - 거래비용은 0 ~ 1 사이의 값으로 지정하면됨(전체 포트폴리오 비중의 0% ~ 100%)

결과
----

* FAA 전략과 동일 가중(Equal weight) 전략에 대한 결과치를 한 눈에 볼 수 있는 그래프   
* FAA 전략과 동일 가중(Equal weight) 전략에 대한 세부 백테스트 결과(pyfolio package 사용)
