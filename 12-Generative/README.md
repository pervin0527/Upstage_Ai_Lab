# 생성형 모델

## 1.배경지식

### 1-1.확률

확률을 이해하기 위해서는 다음과 같은 사전 지식들을 이해해야한다.
- 시행 : 동일한 조건에서 여러 번 반복할 수 있어야하고 그 결과가 우연에 의해 지배되는 실험이나 관찰.
- 표본공간 : 시행 결과들의 **집합**
- 사건 : 표본공간의 **부분집합**
- 근원사건 : 원소의 갯수가 한 개인 사건. 쉽게 말해 원소가 한 개인 표본공간의 부분집합을 말한다.

확률이란 크게 ```수학적 확률```, ```기하학적 확률```, ```통계적 확률```로 구분된다.

#### 수학적 확률 

- 근원사건이 일어날 가능성이 모두 같을 때를 전제로 사용하는 확률.

$$ P(A) = \frac{\emptyset}{n(S)} $$

- 즉, 원소의 개수가 하나인 사건이 일어날 가능성이 모두 같을 때.
- 동전을 던졌을 때 앞면 또는 뒷면이 나올 확률이 같다.
- 사건은 표본공간의 부분집합이므로 공집합이거나 집합 전체일 수도 있다. 이 말은 사건이 절대 발생할 수 없가나 반드시 발생할 수 있음을 말하며 확률이 $ 0 \geq P(A) \geq 1 $이 되는 이유다.

#### 기하학적 확률

- 수학적 확률과 달리 경우의 수나 원소의 개수를 셀 수 없을 경우에 사용하는 확률.
- 양궁 선수가 10점을 맞출 확률은 전체 과녁의 넓이에서 10점 영역의 넓이를 나눈 것과 같다.

#### 통계적 확률

$$ \lim_{n \rightarrow \infty} \frac{r}{n} $$

- 근원사건이 일어날 가능성이 서로 다를 때 사용하는 확률.
- 동전 던지기에서 앞면이 나올 확률과 뒷면이 나올 확률이 서로 다른 경우.
- 극한이 존재한다는 것은 시행의 총 횟수가 무수히 많을 때를 전제로 함을 말한다.

## 2.조건부 확률

$$ P(A | B) = \frac{P(A\cap B)}{P(B)} $$

- 사건 B가 일어났을 때 사건 A가 일어날 확률.
- 즉, 사건 B는 이미 일어난 상태에서 A가 일어날 확률로 표본 공간 S가 B로 바뀌었을 때 사건 A가 일어날 확률과 같다.

## 3.확률변수와 확률분포

여러 번의 시행을 통해 발생한 결과들의 집합이 표본공간.

- 확률변수는 표본공간의 원소들을 실수값으로 대응시키는 함수.
- 확률분포는 각각의 확률변수 값을 확률로 대응시키는 함수.
- 이산적인 확률변수 값을 확률로 대응시키는 함수를 ```확률질량함수```, 연속적인 확률변수 값을 확률로 대응시키는 함수를 ```확률밀도함수```라고 한다.
- 이산확률분포는 이산형 확률변수가 가질 수 있는 모든 값과 그에 대응하는 확률을 종합적으로 나타내는 전체적인 확률 구조.
- 연속확률분포는 연속형 확률변수가 가질 수 있는 모든 값과 그에 대응하는 확률을 종합적으로 나타내는 전체적인 확룰 구조.

## 4.결합분포와 주변분포

- 결합분포는 두 개 이상의 확률 변수가 동시에 특정한 값 또는 범위에 속할 확률을 나타내는 분포. $ P(X, Y) = P(X \cap Y) $
- X가 이미지, Y가 라벨이면 P(X, Y)는 특정 클래스의 이미지가 주어졌을 때 그 이미지가 어떤 특징을 갖는 확률과 어떤 클래스에 속할 확률을 동시에 나타낸다.
- 주변분포는 다변량 확률분포에서 다른 변수는 고려하지 않고 특정 변수 하나로 해당변수의 분포.
- 생성형 모델이나 오토인코더 같은 모델에서 marginal distribution p(x)는 데이터 샘플을 보고 전체 데이터셋의 분포를 추정하겠다는 의미.