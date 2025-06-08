# Facial-Expression-Recognition-Challenge
wandb link: https://wandb.ai/glasa21-free-university-of-tbilisi-/facial-expression-recognition?nw=nwuserglasa21


# Facial Expression Recognition - Iterative CNN Development

## Overview
ამ პროექტში ნაჩვენებია ადამიანის ემოციების ამოცნობის ამოცანა, რომელიც გადაჭრილია CNN არქიტექტურით. დავიწყე მარტივი მოდელით და იტერაციულად გავზარდე კომპლექსურობა წინა მოდელის შედეგებიდან გამომდინარე.

## Dataset
- **Task**: 7-class facial expression recognition
- **Input**: RGB images (128x128 pixels)
- **Classes**: 7 different facial expressions

## Iterative Development Process

### Iteration 1: SimpleCNN (Baseline)
**Architecture Philosophy**: უმარტივესი არქიტექტურა baseline performance-ის მისაღწევად.

```
class SimpleCNN(nn.Module):
    - Conv2d(3 -> 32, 3x3) + ReLU + MaxPool
    - Conv2d(32 -> 64, 3x3) + ReLU + MaxPool  
    - FC(64*32*32 -> 128) + ReLU + Dropout(0.2)
    - FC(128 -> 7)
```

**შემდეგი ეტაპი**: არქიტესტურა არის ძალიან მარტივი, საჭიროა სიღრმის გაზრდა კომპლექსური feature-ებისათვის.

---

### Iteration 2: SimpleCNN_v2
დავამატე 1 conv layer და ასევე BatchNorm layer-ები, გავზარდე მოდელის კომპლექსურობა და სტაბილურობა.

```
class SimpleCNN_v2(nn.Module):
    - Conv2d(3 -> 32) + BatchNorm + ReLU + MaxPool
    - Conv2d(32 -> 64) + BatchNorm + ReLU + MaxPool
    - Conv2d(64 -> 128) + BatchNorm + ReLU + MaxPool  # NEW LAYER
    - FC(128*16*16 -> 256) + ReLU + Dropout(0.3)
    - FC(256 -> 7)
```

**შედეგები**: 
- უკეთესი პერფორმანსი
- უკეთესი training stability

**შემდეგი ეტაპი**: too agressive jumps in parameters

---

### Iteration 3: ImprovedCNN 
32→64→128→128 (less aggressive jumps) + conv layer

```
class ImprovedCNN(nn.Module):
    - Conv2d(3 -> 32) + BatchNorm + ReLU + MaxPool
    - Conv2d(32 -> 64) + BatchNorm + ReLU + MaxPool     
    - Conv2d(64 -> 128) + BatchNorm + ReLU + MaxPool      
    - Conv2d(128 -> 128) + BatchNorm + ReLU + MaxPool   
    - FC(128*8*8 -> 256) + ReLU + Dropout(0.4)
    - FC(256 -> 7)
```

**Results**: 
- **Training**: 67.87%
- **Validation**: 57.11%
- **Overfitting gap**: 10.76%

**შედეგი**: კარგი პერფორმანსი, მაგრამ მაღალი ოვერფიტი, საჭიროა რეგულარიზაცია. 

---

### Iteration 4: EnhancedCNN 
ვცადე ორმაგი conv layer-ები უკეთესი feature extraction-ის მიზნით.
შევამცირე ფილტრები 128->96 და შევცვალე dropout სტრატეგია სხვადასხვა ეტაპებისთვის.

```
class EnhancedCNN(nn.Module):
    - Conv2d(3 -> 32) + ReLU + Conv2d(32 -> 32) + ReLU + BatchNorm + MaxPool
    - Conv2d(32 -> 64) + ReLU + Conv2d(64 -> 64) + ReLU + BatchNorm + MaxPool + Dropout(0.25)
    - Conv2d(64 -> 128) + ReLU + Conv2d(128 -> 128) + ReLU + BatchNorm + MaxPool
    - Conv2d(128 -> 96) + ReLU + Conv2d(96 -> 96) + ReLU + BatchNorm + AdaptiveAvgPool2d(4,4)

    - FC(96*4*4 -> 128) + ReLU + Dropout(0.5)
    - FC(128 -> 64) + ReLU + Dropout(0.3)  
    - FC(64 -> 7)
```

**ჰიპერპარამეტრები**:
- LR: 5e-5 (შევამცირე სტაბილურობის გასაზრდელად)
- Batch Size: 32 (გავორმაგე)
- Weight Decay: 5e-4 (გავზარდე რეგულარიზაციისთვის)
- Epochs: 10

**Results**: 
- **Training**: 71.53%
- **Validation**: 61.54%
- **Overfitting gap**: 9.99%

**შედეგები**: საუკეთესო პერფორმანსი ჯერ-ჯერობით, მაგრამ მაინც მაღალი ოვეფიტი, საჭირო იყო რეგულარიზაციისთვის fine tuning. 

---

### Iteration 5: EnhancedCNN_v2 (Fine-tuned Regularization) - **FINAL MODEL**
დავტოვე იგივე არქიტექტურა, რომელმაც კარგი შედეგი მომცა, უბრალოდ შევცვალე რეგულარიზაციის სტრატეგია

```
class EnhancedCNN_v2(nn.Module):
    # SAME ARCHITECTURE as EnhancedCNN (proven to work well)
    # ONLY CHANGE: Stronger regularization
    
    - dropout1 = nn.Dropout(0.35)  # Increased from 0.25
    - dropout2 = nn.Dropout(0.6)   # Increased from 0.5  
    - dropout3 = nn.Dropout(0.4)   # Increased from 0.3
```

- LR: 4e-5 
- Weight Decay: 8e-4
- Epochs: 15

**Results**: 
- **Training**: 64.53%
- **Validation**: 60.31%
- **Peak Validation**: 61.15% (epoch 14)
- **Overfitting gap**: 4.22%

**შედეგები**: 
- ოვერფიტი შემცირდა საგრძნობლად 9.99% → 4.22% 
- შევინარჩუნე ვალიდაციის კარგი შედეგი (60.31%)

---

## ანალიზი

დავიწყე მარტივი მოდელით და თითოეულ ეტაპზე ვეცადი წინა მოდელის გაუმჯობესება.
აღმოვაჩინე კარგი არქიტექტურა, რომელმაც სასურველი შედეგი დადო, მაგრამ აღმომაჩნდა ოვერფიტი.
ამიტომ, შევცვალე რეგულარიზაციის სტრატეგია და მივაგენი სასურველ ჰიპერპარამეტრებს.
რადგანაც ბოლო მოდელს აღმოაჩნდა კარგი რეგულარიზაცია, გავზარდე ეპოქების რაოდენობაც.
საბოლოო მოდელად ავირჩიე EnhancedCNN_v2, რომელსაც აქვს მისაღები გენერალიზაცია (4.22% overfitting gap), თანაც
ინარჩუნებს კარგ პერფორმანსს ვალიდაციის დროს.

საბოლოოდ იგივე მოდელის თავიდან დატრეინინგება მომიწია, რათა test set-ზე გამეტესტა და მცირედით განსხვავებული შედეგები
ჰქონდა(train acc- 64.4805, val_accuracy	58.8946, test - 58.3237), თუმცა მაინც პირვანდელ შედეგებთან ძალიან ახლოს არის და 
ეს არ იყო პრობლემა.

