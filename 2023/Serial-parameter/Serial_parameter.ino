#include "ser_par.h"

int a = 0;
int b = 0;
int c = 0;
int d = 0;
int e = 0;

int *par_list[5] = {&a,&b,&c,&d,&e};

void setup() {
  Serial.begin(115200);
}

void loop() {
  //do something
  delay(100);

  //Serial_parameter function
  ser_par(par_list);

  //print results
  Serial.printf("a:%d, b:%d, c:%d, d:%d, e:%d\n",a,b,c,d,e);
}
