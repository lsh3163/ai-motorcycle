#include <AFMotor.h>

AF_DCMotor motor1(3);
AF_DCMotor motor2(4);

String Speed;
char  LorR;
int  i, s;

byte DataToRead[6];

void setup() {
  motor1.setSpeed(200);
  motor2.setSpeed(200);

  motor1.run(RELEASE);
  motor2.run(RELEASE);
  
  Serial.begin(9600);
}

void loop() {
  DataToRead[5] = '\n';
  Serial.readBytesUntil(char(13), DataToRead, 5);
  
/* For Debugging, send string to RPi */
  for (i = 0; i < 6; i++) {
    Serial.write(DataToRead[i]);
    if (DataToRead[i] == '\n') break;
  }
/* End of Debugging */

  LorR = DataToRead[0];
  Speed = "";
  for (i = 1; (DataToRead[i] != '\n') && (i < 6); i++) {
    Speed += DataToRead[i];
  }
  s = Speed.toInt();
  if (LorR == 'L') {
    // Turn left wheel with speed s
    motor1.run(FORWARD);
    motor1.setSpeed(s);
  }
  else if (LorR == 'R') {
    // Turn right wheel with speed s  
    motor2.run(FORWARD);
    motor2.setSpeed(s);
  }
  delay(2000); 
}
