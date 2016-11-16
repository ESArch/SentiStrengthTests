import psycopg2

conn = psycopg2.connect("dbname=postgis_22_sample user=postgres password=postgres")
print("Connected to database")

query = "SELECT twe_texto FROM tweet, usuario WHERE twe_usuario = usu_id AND usu_turista AND twe_idioma = 'en'"

cur = conn.cursor()
cur.execute(query)

result = cur.fetchall()
cur.close()
conn.close()
print("Disconnected from database")

print(len(result))

f = open("tweetsTuristas.txt", 'w', encoding='utf8')

tweets = [str(i[0]) for i in result]
for tweet in tweets:
    mystr = '. '.join(tweet.splitlines())
    f.write(mystr+'\n')