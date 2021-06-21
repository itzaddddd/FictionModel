# FictionModel

Fiction Gene Model API

API for predict the genre of the novel

How to install

1. pip install -r requirements.txt

2. py app.py

How to use

1. Use route '/predict' with Method POST 

2. Set Request Header as 'Content-Type' : 'application/json'

3. Send Request with json key 'content'

Example

Request

{
  "content": "สวัสดีทุกท่าน นี่คือตัวอย่างของการทดสอบระบบใบสร้างภาพนิยายตามหลักประเภทรูปแบบใหม่"
}

Response

[{'genre': 'drama','prob' : 0.5212},{'genre': 'romantic','prob' : 0.3211},{'genre': 'thriller','prob' : 0.1099},{...},{...},{...}]
