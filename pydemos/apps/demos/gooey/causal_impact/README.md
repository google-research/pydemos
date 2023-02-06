gcloud builds submit --tag gcr.io/google.com:bigquery-ml-training/causality-gui --project=google.com:bigquery-ml-training

gcloud run deploy --image gcr.io/google.com:bigquery-ml-training/causality-gui --platform managed --google.com:bigquery-ml-training

Cloud Run API credentials
417922191226-omct62fqqgsalg2mqpp3h20q1lshcnhs.apps.googleusercontent.com


curl -H "Authorization: Bearer $(ya29.a0AeTM1ich33xZccQN8baWTHOvTWd9lBT05c-JIEWeu7Mv2D65Bupd8gY8DDDGy7iVxMe4QbTSi8BBPi1rxVMpRP6f4DX1b_pwEZEf_2w18g3ESIZM1yoFhVw9vTckZ-7ZhDYXBKhvobH00flXOjZ98vYyK48X7V55I-eGF52zLeq2saRTZrlfyqg2TV8O0dbk1tpR5zNzWlm7VADdh5XOqEcMiJh00yjjxCJJG1TMdfaKWR_Kdz9zRTJRo3bUK0KM_B9j5v4aCgYKAZUSAQ8SFQHWtWOmD5QXMXMYQ8KMlUMzFHSO_w0270)" <https://default-20221117t143125-5647epmkva-uc.a.run.app>

alias gcurl='curl --header "Authorization: Bearer $(gcloud auth print-identity-token)"'



cp ~/Documents/Data/exci_gui third_party/py/pydemos/apps/demos.
