from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import traceback


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    print("Ala")
    try:
        csv_file = request.files['file']
        if csv_file:
            print('Nazwa pliku:', csv_file.filename)
            print('Typ pliku:', csv_file.content_type)
        df = pd.read_csv(csv_file)
        print(df.head())

        # Poniżej możesz dodać kod generowania wykresu na podstawie danych z pliku CSV
        # Przykład:
        plt.plot([1,2,3],[2,4,6])
        # plt.plot(df['T'], df['val'])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Wykres')
        plt.savefig('static/plot.png')
        plt.close()

        return render_template('result.html')
    except Exception as e:
        traceback.print_exc()
        return 'Wystąpił błąd: ' + str(e)

if __name__ == '__main__':
    app.run()
