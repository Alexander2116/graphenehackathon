int bias = A0; // Brown
int read_1 = A1; // Red
int read_2 = A2; // Orange
int read_3 = A3; // Yellow

double value_1 = 0.0;
double value_2 = 0.0;
double value_3 = 0.0;

int bias_val = 1023.0/5.0;

double v12 = 0.0;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  pinMode(A1, INPUT);
  pinMode(A2, INPUT);
  pinMode(A3, INPUT);
}

void loop() {
  // put your main code here, to run repeatedly:
  value_1 = (analogRead(A1)*5.0)/1023.0;
  value_2 = (analogRead(A2)*5.0)/1023.0;
  value_3 = (analogRead(A3)*5.0)/1023.0;
  Serial.println(value_1,7); 
  delay(100);
}
